"""Sample TRAINING MANIFEST builder for the Qwen3-VL gaze profile (gaze-8yc.4).

Turns the per-episode extraction bundles + recipes into training EXAMPLES for the
canonical profile ``qwen3-vl-gaze-5hz-392px`` (see
``docs/qwen3_vl_conversion_settings.md``):

    input:  a short clip of video frames sampled on the canonical 5 Hz timeline
    target: a single normalized 2D gaze point g_t = [x_norm, y_norm] in [0, 1]

This module is the training-manifest sibling of ``smoke.py`` / ``overlay.py``. It
REUSES the validated curation infrastructure rather than re-deriving anything:

  * ``curate_readers.extract_episode`` / pre-extracted ``/tmp/gaze_extract/<slug>_full.json``
    bundles for per-episode gaze rows + annotation segments + video metadata.
  * ``overlay.build_episode_data`` -> :class:`overlay.EpisodeData`, then
    ``overlay.reconciled_gaze_times`` / ``overlay.reconciled_annotations`` for the
    epoch_sync clock reconciliation onto the common video-zero-seconds clock.
  * ``overlay.build_projection_context`` + ``overlay.project_one`` for the validated
    per-sample gaze -> source-mp4-pixel projection (loads calibration ONCE).
  * ``video.transcode_video`` semantics for the 392x392 pad/h264/5Hz resample.

------------------------------------------------------------------------------
DESIGN DECISIONS  (documented here + echoed in the validation report; flagged for
human review, see FLAGS below -- none of them block the sample build).
------------------------------------------------------------------------------

1. CLIP TEMPORALITY  -- default ``causal`` past-context.
   The doc says the canonical training form is ``frames=[t, t+1/fps, ... t+(n-1)/fps]``
   but notes the future-context form is *offline only* (frames after t can carry
   information unavailable at t). We default to CAUSAL past-context
   ``frames=[t-(n-1)/fps, ..., t], label=gaze[t]`` because the stated model goal is
   online/causal gaze prediction. ``--temporality {causal,centered,future}`` switches it.
   FLAG: confirm the production run wants causal (vs the doc's literal future form).

2. ANCHOR GRID + STRIDE.
   Anchors live on the canonical 5 Hz grid. Default ``--stride 8`` frames (=1.6 s
   hop at 5 Hz) -> ~50% overlap for 16-frame clips. The first valid anchor is the one
   whose whole clip window fits inside the video (causal: anchor >= (n-1)/fps; future:
   anchor + (n-1)/fps <= duration). Partial clips at episode edges are DROPPED (we do
   not pad with black/duplicate frames -- that would inject fake context).
   FLAG: stride/overlap is a knob; sweep it in the full run if data volume matters.

3. GAZE TARGET NORMALIZATION -- normalized to the PADDED 392x392 frame.
   The model sees the padded square video, so the [0,1] target must be meaningful on
   THAT frame, not the original. We project gaze to source-mp4 pixels (reusing the
   validated projection), then push it through the SAME pad transform ffmpeg applies
   (scale-to-fit + centre-pad) and divide by the square side. This keeps the dot on
   the object across heterogeneous source resolutions. We also emit the 0-1000 integer
   form (``x_1000 = round(x_norm*1000)`` clamped 0..1000) that Qwen-family models
   commonly bin coordinates to.

4. MASKING / VALIDITY -- keep-with-flag, drop only when the anchor gaze is missing.
   Each example carries ``target_valid``: True when the anchor gaze projected to an
   in-frame [0,1] point, False when it was out-of-frame / behind-camera / invalid
   (we still clamp the stored x_norm/y_norm into [0,1] so downstream code never sees a
   raw out-of-range value, and record ``target_in_frame`` separately). Examples whose
   anchor gaze sample is entirely MISSING (no nearby sample within one gaze period, or
   the projection returned None) are DROPPED -- there is no label to learn.
   FLAG: confirm "keep out-of-frame with target_valid=False" vs "drop" for the full run.

5. ANNOTATION UNIFICATION -- one schema for all 7 datasets.
   Every dataset's channels are flattened to a single list of timed text spans
   ``{start_s, end_s|null, point_s|null, text, channel, kind}`` already reconciled to
   the video clock. For each clip we attach the text active at the anchor time using
   the profile's ``active_interval`` sample_mode: for ``interval`` channels, every span
   covering the anchor; for ``point`` channels, the nearest point within a tolerance.
   Channel provenance is preserved. See :func:`active_annotation_at`.

The per-example schema is documented in :data:`TRAINING_EXAMPLE_SCHEMA` and written
to ``schema.json`` next to the manifest.
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import overlay as ov
from .curate import Puller
from .overlay import (
    EpisodeData,
    active_segments,
    build_episode_data,
    build_projection_context,
    project_one,
    reconciled_annotations,
    reconciled_gaze_times,
)


# --------------------------------------------------------------------------- #
# Profile defaults (mirror docs/qwen3_vl_conversion_settings.md).
# --------------------------------------------------------------------------- #
DEFAULT_PROFILE = "qwen3-vl-gaze-5hz-392px"
DEFAULT_FPS = 5.0
DEFAULT_NUM_FRAMES = 16
DEFAULT_STRIDE = 8
DEFAULT_RESOLUTION = 392
DEFAULT_TEMPORALITY = "causal"

# The integer bin range Qwen-family models commonly map normalized coords onto.
COORD_BINS = 1000

# When picking the gaze sample nearest an anchor time, reject it if no sample lies
# within this many seconds (so we never attach a stale label across a gaze gap).
NEAREST_GAZE_MAX_DT_S = 0.2

# Point-annotation tolerance (seconds) for active_interval lookup on point channels.
POINT_ANNOTATION_TOL_S = 0.5


TRAINING_EXAMPLE_SCHEMA: dict[str, Any] = {
    "$comment": "One JSONL row per training example for the qwen3-vl gaze profile.",
    "fields": {
        "id": "stable example id: <dataset>:<episode_id>#<anchor_frame_index>",
        "dataset": "dataset slug (recipe name)",
        "episode_id": "source episode id",
        "profile": "canonical profile name",
        "video": "relative path (from manifest root) to the per-episode resampled mp4",
        "fps": "frames per second of the resampled video / clip sampling grid",
        "num_frames": "number of frames in the clip",
        "temporality": "causal | centered | future (clip window relative to the anchor)",
        "anchor_frame_index": "0-based frame index of the anchor (label) frame in the resampled video",
        "anchor_time_s": "anchor time in seconds on the video-zero clock (== anchor_frame_index/fps)",
        "frame_indices": "list[int] clip frame indices into the resampled video (monotonic)",
        "frame_times_s": "list[float] clip frame times in seconds (== index/fps)",
        "target": "[x_norm, y_norm] gaze in [0,1] on the PADDED resolutionXresolution frame",
        "target_1000": "[x_1000, y_1000] integer 0..1000 bin form (round(norm*1000), clamped)",
        "target_valid": "bool: anchor gaze projected to an in-frame, in-[0,1] point",
        "target_in_frame": "bool: gaze fell inside the (unpadded) content region",
        "gaze_source_px": "[x_px, y_px] projected gaze in source-mp4 pixels (provenance)",
        "annotation": "active text span at the anchor (active_interval), or null",
        "annotations_all": "list of every active span across channels at the anchor",
        "annotation_text": "convenience: the chosen annotation's text (or '')",
    },
    "annotation_span": {
        "start_s": "interval start on the video clock (null for pure points)",
        "end_s": "interval end on the video clock (null for open/point)",
        "point_s": "point event time on the video clock (null for intervals)",
        "text": "annotation text",
        "channel": "source channel name (provenance)",
        "kind": "interval | point",
    },
    "target_normalization": {
        "space": "padded square frame (resolution x resolution)",
        "method": "project gaze->source px, apply ffmpeg scale-to-fit + centre-pad, /side",
        "bins": COORD_BINS,
    },
}


# =========================================================================== #
# Pad-transform geometry (matches ffmpeg scale=...:force_original_aspect_ratio
# =decrease,pad=side:side:(ow-iw)/2:(oh-ih)/2).
# =========================================================================== #
@dataclass
class PadTransform:
    """Maps a source-frame pixel onto the padded square frame, then to [0,1]."""

    src_w: int
    src_h: int
    side: int

    @property
    def scale(self) -> float:
        return min(self.side / self.src_w, self.side / self.src_h)

    @property
    def content_w(self) -> float:
        return self.src_w * self.scale

    @property
    def content_h(self) -> float:
        return self.src_h * self.scale

    @property
    def offset_x(self) -> float:
        return (self.side - self.content_w) / 2.0

    @property
    def offset_y(self) -> float:
        return (self.side - self.content_h) / 2.0

    def px_to_norm(self, x_px: float, y_px: float) -> tuple[float, float]:
        """Source pixel -> normalized [0,1] coordinate on the padded square frame."""
        px = self.offset_x + x_px * self.scale
        py = self.offset_y + y_px * self.scale
        return px / self.side, py / self.side


def clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def to_bins(norm: float, bins: int = COORD_BINS) -> int:
    """Normalized [0,1] -> integer 0..bins bin (clamped)."""
    b = int(round(clamp01(norm) * bins))
    return 0 if b < 0 else (bins if b > bins else b)


# =========================================================================== #
# Linear resampling of the (reconciled) gaze track onto the canonical fps grid.
# =========================================================================== #
@dataclass
class ResampledGaze:
    """Gaze track resampled onto a uniform fps grid in source-mp4 pixel space.

    ``px``/``py`` are projected source pixels per grid time; ``valid`` marks grid
    points that were produced from at least one nearby real sample (vs extrapolated
    past the ends / across a long gap).
    """

    fps: float
    times_s: list[float]
    px: list[float | None]
    py: list[float | None]
    in_frame: list[bool]
    valid: list[bool]


def _lerp(a: float, b: float, frac: float) -> float:
    return a + (b - a) * frac


def resample_track_linear(
    times: list[float],
    xs: list[float],
    ys: list[float],
    in_frame: list[bool],
    *,
    fps: float,
    duration_s: float,
    max_gap_s: float,
) -> ResampledGaze:
    """Linearly interpolate an irregular (time, x, y) track onto a uniform fps grid.

    ``times`` must be ascending. Grid times run ``0, 1/fps, ...`` up to ``duration_s``.
    A grid point is ``valid=False`` (and its px/py None) when it falls before the first
    / after the last real sample, or inside a gap wider than ``max_gap_s`` (we don't
    invent a label across a long blink/dropout). ``in_frame`` of a grid point is the
    OR of its bracketing samples' in-frame flags (conservative: a target on the edge
    of the FOV stays usable).
    """
    n_grid = int(math.floor(duration_s * fps)) + 1
    grid_t = [i / fps for i in range(n_grid)]
    gx: list[float | None] = [None] * n_grid
    gy: list[float | None] = [None] * n_grid
    gin: list[bool] = [False] * n_grid
    gvalid: list[bool] = [False] * n_grid

    if len(times) == 0:
        return ResampledGaze(fps, grid_t, gx, gy, gin, gvalid)

    j = 0  # pointer into the source track; track is ascending
    n_src = len(times)
    for i, t in enumerate(grid_t):
        # advance j so that times[j] <= t < times[j+1] (or j at an end)
        while j + 1 < n_src and times[j + 1] <= t:
            j += 1
        if t < times[0] or t > times[-1]:
            continue  # outside the track -> stays invalid (no extrapolation)
        if abs(times[j] - t) < 1e-9:
            gx[i], gy[i], gin[i], gvalid[i] = xs[j], ys[j], in_frame[j], True
            continue
        if j + 1 >= n_src:
            # exactly at / past the last sample
            gx[i], gy[i], gin[i], gvalid[i] = xs[-1], ys[-1], in_frame[-1], True
            continue
        t0, t1 = times[j], times[j + 1]
        if (t1 - t0) > max_gap_s:
            continue  # gap too wide -> don't interpolate across it
        frac = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
        gx[i] = _lerp(xs[j], xs[j + 1], frac)
        gy[i] = _lerp(ys[j], ys[j + 1], frac)
        gin[i] = bool(in_frame[j] or in_frame[j + 1])
        gvalid[i] = True
    return ResampledGaze(fps, grid_t, gx, gy, gin, gvalid)


# =========================================================================== #
# Projected gaze track in source pixels (reuses the validated overlay projection).
# =========================================================================== #
def projected_gaze_track(data: EpisodeData, puller: Puller) -> tuple[list[float], list[float], list[float], list[bool]]:
    """Project every gaze sample to source-mp4 pixels on the video clock.

    Returns ascending ``(times_s, xs_px, ys_px, in_frame)`` keeping only samples
    that reconciled to a video-clock time AND projected to a pixel. Reuses
    ``overlay.build_projection_context`` (loads calibration once) + ``project_one``.
    """
    gaze_times = reconciled_gaze_times(data)
    ctx = build_projection_context(data, puller)
    w = data.video.get("width")
    h = data.video.get("height")
    triples: list[tuple[float, float, float, bool]] = []
    for row, tv in zip(data.gaze_rows, gaze_times):
        if tv is None:
            continue
        proj = project_one(row, ctx)
        if proj is None:
            continue
        x_px, y_px = proj
        inf = bool(w and h and 0 <= x_px <= w and 0 <= y_px <= h)
        triples.append((tv, x_px, y_px, inf))
    triples.sort(key=lambda r: r[0])
    times = [r[0] for r in triples]
    xs = [r[1] for r in triples]
    ys = [r[2] for r in triples]
    inf = [r[3] for r in triples]
    return times, xs, ys, inf


# =========================================================================== #
# Annotation unification + active-at-anchor lookup (active_interval sample_mode).
# =========================================================================== #
def unified_annotation_spans(data: EpisodeData) -> list[dict[str, Any]]:
    """Flatten every reconciled channel into one list of timed text spans.

    Single schema across all 7 datasets: ``{start_s, end_s, point_s, text, channel, kind}``
    with times already on the video-zero clock. Channel provenance is kept.
    """
    out: list[dict[str, Any]] = []
    for ch in reconciled_annotations(data):
        name, kind = ch["name"], ch["kind"]
        for s in ch["segments"]:
            text = s.get("text")
            if text is None or str(text).strip() == "":
                # keep timing provenance only when there is some text to attach
                continue
            out.append({
                "start_s": s.get("start_s"),
                "end_s": s.get("end_s"),
                "point_s": s.get("point_s"),
                "text": str(text),
                "channel": name,
                "kind": kind,
            })
    out.sort(key=lambda r: (
        r["start_s"] if r["start_s"] is not None
        else (r["point_s"] if r["point_s"] is not None else math.inf)
    ))
    return out


def active_annotation_at(
    spans: list[dict[str, Any]],
    t: float,
    *,
    point_tol: float = POINT_ANNOTATION_TOL_S,
) -> list[dict[str, Any]]:
    """Spans active at anchor time ``t`` under the ``active_interval`` sample_mode.

    Groups spans by (channel, kind) and applies ``overlay.active_segments`` per channel
    (interval: every covering span; point: nearest within ``point_tol``). Preserves the
    channel/kind fields so every returned span keeps its provenance.
    """
    by_channel: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for s in spans:
        by_channel.setdefault((s["channel"], s["kind"]), []).append(s)
    active: list[dict[str, Any]] = []
    for (channel, kind), segs in by_channel.items():
        for s in active_segments(segs, t, kind=kind, point_tol=point_tol):
            active.append(s)
    # Order intervals first (more specific), then by start/point time.
    active.sort(key=lambda r: (
        r["kind"] != "interval",
        r["start_s"] if r["start_s"] is not None
        else (r["point_s"] if r["point_s"] is not None else math.inf),
    ))
    return active


# =========================================================================== #
# Resampled per-episode video (392x392 pad / 5 Hz / h264 mp4).
# =========================================================================== #
def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def resample_episode_video(
    data: EpisodeData,
    puller: Puller,
    out_path: Path,
    *,
    fps: float,
    side: int,
    window_s: float | None,
    start_s: float | None = None,
    spans: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Materialize the per-episode sample clip at ``side``x``side`` / ``fps`` / h264 mp4.

    For huge source mp4s (nymeria/hd-epic/holoassist/ego-exo4d) we pull then trim to a
    short ``window_s`` clip first (reusing ``overlay.pull_and_trim`` which deletes the
    big source after the trim), keeping remote work light. Small sources (egome/egtea)
    are transcoded directly. The training clips index into THIS resampled video.

    When ``start_s`` is None and ``window_s`` is set, the window is auto-placed to
    overlap annotation activity (reusing ``overlay._auto_window`` over the reconciled
    ``spans``), so a sample clip from a long episode (e.g. nymeria, whose annotations
    begin ~98 s in) actually contains annotated frames instead of an empty head window.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    workdir = out_path.parent / f"_work_{data.slug}"
    workdir.mkdir(parents=True, exist_ok=True)

    vid_dur = float(data.video.get("duration_s") or 0.0)
    if start_s is None:
        if window_s is not None and spans:
            # ov._auto_window wants channels with .segments using start_s/point_s
            annos = [{"name": "_", "kind": "interval", "segments": spans}]
            start_s = ov._auto_window(annos, vid_dur, window_s)
        else:
            start_s = 0.0
    span = vid_dur if window_s is None else min(window_s, max(0.5, vid_dur - start_s))

    # Pull + trim to the window (deletes the big source after).
    src_clip = ov.pull_and_trim(
        data, puller, start_s=start_s, max_seconds=span, workdir=workdir
    )

    # Resample: scale-to-fit + centre-pad to side x side, drop audio, h264, fps.
    vf = (
        f"fps={fps},"
        f"scale={side}:{side}:force_original_aspect_ratio=decrease,"
        f"pad={side}:{side}:(ow-iw)/2:(oh-ih)/2"
    )
    cmd = [
        "ffmpeg", "-y", "-i", str(src_clip), "-vf", vf, "-an",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path),
    ]
    r = _run(cmd)
    if r.returncode != 0 or not out_path.exists():
        raise RuntimeError(f"resample failed for {data.slug}: {r.stderr[-800:]}")

    meta = ov._probe(out_path)
    # cleanup the work dir (trimmed clip + frames)
    try:
        shutil.rmtree(workdir)
    except OSError:
        pass
    return {
        "path": str(out_path),
        "width": meta["width"],
        "height": meta["height"],
        "fps": fps,
        "duration_s": meta.get("duration"),
        "window_s": span,
        "start_s": start_s,
    }


# =========================================================================== #
# Sliding-window clip construction.
# =========================================================================== #
def clip_frame_indices(
    anchor_idx: int, num_frames: int, *, temporality: str
) -> list[int]:
    """Frame indices for a clip anchored at ``anchor_idx`` (the label frame).

    causal:   [anchor-(n-1), ..., anchor]                 (anchor is the LAST frame)
    centered: anchor centered (anchor is the middle frame for odd n)
    future:   [anchor, ..., anchor+(n-1)]                 (anchor is the FIRST frame)
    """
    if temporality == "causal":
        return list(range(anchor_idx - (num_frames - 1), anchor_idx + 1))
    if temporality == "future":
        return list(range(anchor_idx, anchor_idx + num_frames))
    if temporality == "centered":
        half = num_frames // 2
        start = anchor_idx - half
        return list(range(start, start + num_frames))
    raise ValueError(f"unknown temporality {temporality!r}")


def build_episode_examples(
    data: EpisodeData,
    resampled: ResampledGaze,
    spans: list[dict[str, Any]],
    pad: PadTransform,
    *,
    video_rel: str,
    fps: float,
    num_frames: int,
    stride: int,
    temporality: str,
    n_video_frames: int,
) -> list[dict[str, Any]]:
    """Build sliding-window training examples for one episode.

    Anchors march across the 5 Hz grid by ``stride``; an anchor is kept only when its
    whole clip window lands inside ``[0, n_video_frames-1]`` (drop partial edges) and the
    resampled anchor gaze sample exists. Out-of-frame/invalid anchors are kept with
    ``target_valid=False`` (only entirely-missing anchors are dropped).
    """
    examples: list[dict[str, Any]] = []
    grid_n = min(len(resampled.times_s), n_video_frames)

    for anchor_idx in range(0, grid_n, stride):
        frames = clip_frame_indices(anchor_idx, num_frames, temporality=temporality)
        if frames[0] < 0 or frames[-1] > grid_n - 1:
            continue  # partial clip at an episode edge -> drop

        # Anchor gaze: must exist (else there is no label to learn).
        ax, ay = resampled.px[anchor_idx], resampled.py[anchor_idx]
        if ax is None or ay is None or not resampled.valid[anchor_idx]:
            continue

        x_norm_raw, y_norm_raw = pad.px_to_norm(ax, ay)
        in_frame = bool(resampled.in_frame[anchor_idx])
        target_valid = bool(in_frame and 0.0 <= x_norm_raw <= 1.0 and 0.0 <= y_norm_raw <= 1.0)
        x_norm = clamp01(x_norm_raw)
        y_norm = clamp01(y_norm_raw)

        anchor_time = anchor_idx / fps
        active = active_annotation_at(spans, anchor_time)
        chosen = active[0] if active else None

        examples.append({
            "id": f"{data.slug}:{data.episode_id}#{anchor_idx}",
            "dataset": data.slug,
            "episode_id": data.episode_id,
            "profile": DEFAULT_PROFILE,
            "video": video_rel,
            "fps": fps,
            "num_frames": num_frames,
            "temporality": temporality,
            "anchor_frame_index": anchor_idx,
            "anchor_time_s": round(anchor_time, 6),
            "frame_indices": frames,
            "frame_times_s": [round(f / fps, 6) for f in frames],
            "target": [round(x_norm, 6), round(y_norm, 6)],
            "target_1000": [to_bins(x_norm), to_bins(y_norm)],
            "target_valid": target_valid,
            "target_in_frame": in_frame,
            "gaze_source_px": [round(ax, 3), round(ay, 3)],
            "annotation": chosen,
            "annotations_all": active,
            "annotation_text": (chosen["text"] if chosen else ""),
        })
    return examples


# =========================================================================== #
# Top-level builder.
# =========================================================================== #
def build_training_manifest(
    out_root: str | Path,
    *,
    datasets: list[str] | None = None,
    episodes: dict[str, str] | None = None,
    sample_extra: dict[str, dict[str, Any]] | None = None,
    profile: str = DEFAULT_PROFILE,
    fps: float = DEFAULT_FPS,
    num_frames: int = DEFAULT_NUM_FRAMES,
    stride: int = DEFAULT_STRIDE,
    resolution: int = DEFAULT_RESOLUTION,
    temporality: str = DEFAULT_TEMPORALITY,
    window_s: float | None = 30.0,
    puller: Puller | None = None,
    reuse_bundle: bool = True,
) -> dict[str, Any]:
    """Build the sample training manifest for one episode per selected dataset.

    Writes ``<out_root>/manifest.jsonl`` (one row per example),
    ``<out_root>/manifest.parquet[.jsonl]``, ``<out_root>/schema.json``,
    ``<out_root>/training_report.json``, and per-episode resampled mp4s under
    ``<out_root>/videos/<dataset>/<episode>.mp4``.
    """
    out_root = Path(out_root)
    (out_root / "videos").mkdir(parents=True, exist_ok=True)
    puller = puller or Puller()

    sample_map = _sample_episode_map()
    if episodes:
        sample_map.update({k: {"episode_id": v} for k, v in episodes.items()})
    for slug, extra in (sample_extra or {}).items():
        sample_map.setdefault(slug, {}).update(extra)

    slugs = sorted(sample_map.keys())
    if datasets:
        want = set(datasets)
        slugs = [s for s in slugs if s in want]

    all_examples: list[dict[str, Any]] = []
    rows_report: list[dict[str, Any]] = []

    for slug in slugs:
        entry = sample_map[slug]
        episode_id = entry["episode_id"]
        extra = {k: v for k, v in entry.items() if k != "episode_id"}
        try:
            data = build_episode_data(slug, episode_id, puller, sample_extra=extra, reuse=reuse_bundle)
        except Exception as exc:  # noqa
            rows_report.append({"dataset": slug, "episode": episode_id, "error": str(exc)})
            continue

        src_w = int(data.video.get("width") or 0)
        src_h = int(data.video.get("height") or 0)

        # 1. unified annotations on the video clock (used both to auto-place the
        #    sample window and, after rebasing, to label clips).
        spans_video = unified_annotation_spans(data)

        # 2. resampled per-episode video (resolution^2 pad / fps / h264). When a window
        #    is set, auto-place it to overlap annotation activity.
        video_rel = f"videos/{slug}/{_safe(episode_id)}.mp4"
        try:
            vmeta = resample_episode_video(
                data, puller, out_root / video_rel,
                fps=fps, side=resolution, window_s=window_s, spans=spans_video,
            )
        except Exception as exc:  # noqa
            rows_report.append({"dataset": slug, "episode": episode_id, "error": f"video: {exc}"})
            continue

        clip_dur = float(vmeta.get("duration_s") or vmeta.get("window_s") or 0.0)
        n_video_frames = int(math.floor(clip_dur * fps)) + 1
        # the resampled clip starts at vmeta["start_s"] of the original; clip-local clock
        clip_start_offset = float(vmeta.get("start_s") or 0.0)

        # 3. projected gaze track (source px, video clock), rebased to clip-local time
        times, xs, ys, inf = projected_gaze_track(data, puller)
        # restrict to the resampled window and rebase to clip-local seconds
        rebased = [
            (t - clip_start_offset, x, y, f)
            for t, x, y, f in zip(times, xs, ys, inf)
            if clip_start_offset - 1e-6 <= t <= clip_start_offset + clip_dur + 1e-6
        ]
        r_times = [r[0] for r in rebased]
        r_xs = [r[1] for r in rebased]
        r_ys = [r[2] for r in rebased]
        r_inf = [r[3] for r in rebased]
        max_gap_s = 1.0  # don't interpolate gaze across gaps > 1 s
        resampled = resample_track_linear(
            r_times, r_xs, r_ys, r_inf, fps=fps, duration_s=clip_dur, max_gap_s=max_gap_s,
        )

        # 4. rebase the unified annotation spans to clip-local seconds
        spans = [dict(s) for s in spans_video]
        for s in spans:
            for k in ("start_s", "end_s", "point_s"):
                if s[k] is not None:
                    s[k] = s[k] - clip_start_offset

        # 5. sliding-window clips
        pad = PadTransform(src_w=src_w or resolution, src_h=src_h or resolution, side=resolution)
        examples = build_episode_examples(
            data, resampled, spans, pad,
            video_rel=video_rel, fps=fps, num_frames=num_frames, stride=stride,
            temporality=temporality, n_video_frames=n_video_frames,
        )
        all_examples.extend(examples)

        # 6. per-episode report row
        n_clips = len(examples)
        n_valid = sum(1 for e in examples if e["target_valid"])
        n_with_anno = sum(1 for e in examples if e["annotations_all"])
        total_spans = len(spans)
        rows_report.append({
            "dataset": slug,
            "episode": episode_id,
            "num_clips": n_clips,
            "target_valid_clips": n_valid,
            "target_coverage_pct": round(100.0 * n_valid / n_clips, 1) if n_clips else 0.0,
            "clips_with_annotation": n_with_anno,
            "annotation_spans": total_spans,
            "video_width": vmeta["width"],
            "video_height": vmeta["height"],
            "video_fps": vmeta["fps"],
            "video_duration_s": round(clip_dur, 3),
            "window_start_s": round(clip_start_offset, 3),
            "source_dims": f"{src_w}x{src_h}",
            "projection": data.projection_method,
            "gaze_space": data.gaze_space,
        })

    # write outputs
    _write_jsonl(all_examples, out_root / "manifest.jsonl")
    _write_table(all_examples, out_root / "manifest.parquet")
    (out_root / "schema.json").write_text(
        json.dumps(_full_schema(profile, fps, num_frames, stride, resolution, temporality),
                   indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report = {
        "profile": profile,
        "fps": fps,
        "num_frames": num_frames,
        "stride": stride,
        "resolution": resolution,
        "temporality": temporality,
        "window_s": window_s,
        "total_examples": len(all_examples),
        "episodes": rows_report,
    }
    (out_root / "training_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _sample_episode_map() -> dict[str, dict[str, Any]]:
    path = Path(__file__).resolve().parents[2] / "recipes" / "_sample_episodes.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for slug, entry in (data.get("samples") or {}).items():
        e = {k: v for k, v in entry.items() if k != "note"}
        out[slug] = e
    return out


def _full_schema(profile, fps, num_frames, stride, resolution, temporality) -> dict[str, Any]:
    return {
        "profile": profile,
        "fps": fps,
        "num_frames": num_frames,
        "stride": stride,
        "resolution": resolution,
        "temporality": temporality,
        "example": TRAINING_EXAMPLE_SCHEMA,
    }


def _safe(name: str) -> str:
    return name.replace("/", "__").replace(":", "_")


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")


def _write_table(rows: list[dict[str, Any]], path: Path) -> None:
    from .table import write_table

    # parquet can't store ragged lists/dicts cleanly; thin to scalar columns for the
    # table form and keep the full nested record in manifest.jsonl.
    flat = []
    for r in rows:
        flat.append({
            "id": r["id"],
            "dataset": r["dataset"],
            "episode_id": r["episode_id"],
            "anchor_time_s": r["anchor_time_s"],
            "num_frames": r["num_frames"],
            "fps": r["fps"],
            "x_norm": r["target"][0],
            "y_norm": r["target"][1],
            "x_1000": r["target_1000"][0],
            "y_1000": r["target_1000"][1],
            "target_valid": r["target_valid"],
            "annotation_text": r["annotation_text"],
        })
    write_table(flat, path)
