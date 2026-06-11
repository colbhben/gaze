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

from . import molmoact as ma
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
# PadTransform / clamp01 / to_bins / COORD_BINS live in curate.py (the contract base)
# so molmoact.py can reuse them without a circular import. Re-exported here for
# back-compat (tests and callers import them from training).
from .curate import PadTransform, clamp01, to_bins  # noqa: E402,F401


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
def projected_gaze_track(
    data: EpisodeData, puller: Puller, *, project_max_hz: float | None = None,
) -> tuple[list[float], list[float], list[float], list[bool]]:
    """Project gaze samples to source-mp4 pixels on the video clock.

    Returns ascending ``(times_s, xs_px, ys_px, in_frame)`` keeping only samples
    that reconciled to a video-clock time AND projected to a pixel. Reuses
    ``overlay.build_projection_context`` (loads calibration once) + ``project_one``.

    Projection can be EXPENSIVE for Aria CPF (projectaria_tools per-sample: ~60s for
    a 30k-sample nymeria take). Since the projected track is only ever linearly
    resampled onto a low-fps grid downstream, ``project_max_hz`` subsamples the raw
    rows by time to at most that rate BEFORE projecting (e.g. 12 Hz for a 5 fps grid
    -> ~5x fewer projections, accurate interpolation preserved). None = project all.
    """
    gaze_times = reconciled_gaze_times(data)
    ctx = build_projection_context(data, puller)
    w = data.video.get("width")
    h = data.video.get("height")
    min_dt = (1.0 / project_max_hz) if project_max_hz else 0.0
    triples: list[tuple[float, float, float, bool]] = []
    # rows are time-ordered after reconciliation; subsample by time before projecting.
    pairs = sorted(
        ((tv, row) for row, tv in zip(data.gaze_rows, gaze_times) if tv is not None),
        key=lambda p: p[0],
    )
    last_t = None
    for tv, row in pairs:
        if min_dt and last_t is not None and (tv - last_t) < min_dt:
            continue
        proj = project_one(row, ctx)
        if proj is None:
            continue
        last_t = tv
        x_px, y_px = proj
        inf = bool(w and h and 0 <= x_px <= w and 0 <= y_px <= h)
        triples.append((tv, x_px, y_px, inf))
    times = [r[0] for r in triples]
    xs = [r[1] for r in triples]
    ys = [r[2] for r in triples]
    inf = [r[3] for r in triples]
    return times, xs, ys, inf


# =========================================================================== #
# Gaze validity-gap cull (dataset_filters.max_gaze_valid_gap_s, e.g. egome 0.25).
# =========================================================================== #
def check_gaze_validity_gaps(data: EpisodeData, max_gap_s: float) -> tuple[bool, float]:
    """Return (passes, observed_max_gap_s) for the reconciled gaze track.

    ``passes`` is False (episode should be culled) when the time gap between any two
    consecutive VALID gaze samples exceeds ``max_gap_s``. Reuses the same valid flag
    the overlay uses; times are reconciled to the video-zero clock.
    """
    from . import curate_readers as cr

    gaze_times = reconciled_gaze_times(data)
    valid = [bool(r.get("valid", True)) for r in data.gaze_rows]
    exceeded, observed = cr.max_gaze_gap_exceeded(gaze_times, valid, max_gap_s)
    return (not exceeded, observed)


# =========================================================================== #
# Annotation-bounded clip chopping (max-duration + merge + min-length).
# =========================================================================== #
def chop_into_segments(
    spans: list[dict[str, Any]],
    *,
    max_clip_s: float,
    merge_gap_s: float = 1.0,
    drop_shorter_than_s: float = 0.5,
    duration_s: float | None = None,
) -> list[dict[str, Any]]:
    """Chop an episode into annotation-bounded clip segments on the video clock.

    Given unified annotation spans (``unified_annotation_spans``, video-zero clock),
    produce a list of ``{start_s, end_s}`` segments such that:

      * only regions WITH annotation activity become segments (uninteresting/empty
        regions are dropped),
      * spans separated by <= ``merge_gap_s`` are merged into one contiguous region,
      * any merged region longer than ``max_clip_s`` is split into <= ``max_clip_s``
        sub-segments, preferring to cut at an interior span boundary near the cut point
        (falling back to a hard cut at ``max_clip_s``),
      * segments shorter than ``drop_shorter_than_s`` are dropped.

    ``duration_s`` (video length) clamps the final segment end. Point annotations
    (no end) contribute a [point - drop_shorter_than_s/2, point + drop_shorter_than_s/2] sliver so they
    are not lost.
    """
    # 1. Normalize each span to an [a, b] interval on the video clock.
    intervals: list[tuple[float, float]] = []
    boundaries: set[float] = set()
    half = drop_shorter_than_s / 2.0
    for s in spans:
        a = s.get("start_s")
        b = s.get("end_s")
        p = s.get("point_s")
        if a is not None and b is not None and b > a:
            lo, hi = float(a), float(b)
        elif p is not None:
            lo, hi = float(p) - half, float(p) + half
        elif a is not None:
            lo, hi = float(a), float(a) + drop_shorter_than_s
        else:
            continue
        if duration_s is not None:
            lo = max(0.0, lo)
            hi = min(float(duration_s), hi)
        if hi <= lo:
            continue
        intervals.append((lo, hi))
        boundaries.add(lo)
        boundaries.add(hi)
    if not intervals:
        return []

    # 2. Merge overlapping / close (<= merge_gap_s) intervals into regions.
    intervals.sort()
    regions: list[tuple[float, float]] = []
    cur_lo, cur_hi = intervals[0]
    for lo, hi in intervals[1:]:
        if lo - cur_hi <= merge_gap_s:
            cur_hi = max(cur_hi, hi)
        else:
            regions.append((cur_lo, cur_hi))
            cur_lo, cur_hi = lo, hi
    regions.append((cur_lo, cur_hi))

    # 3. Split long regions at <= max_clip_s, preferring an interior span boundary.
    sorted_bounds = sorted(boundaries)
    segments: list[dict[str, Any]] = []
    for lo, hi in regions:
        start = lo
        while hi - start > max_clip_s + 1e-6:
            hard_cut = start + max_clip_s
            # prefer the latest span boundary in (start, hard_cut] that leaves a
            # segment >= drop_shorter_than_s; else hard cut.
            candidates = [b for b in sorted_bounds if start + drop_shorter_than_s <= b <= hard_cut]
            cut = candidates[-1] if candidates else hard_cut
            if cut - start >= drop_shorter_than_s:
                segments.append({"start_s": round(start, 6), "end_s": round(cut, 6)})
            start = cut
        if hi - start >= drop_shorter_than_s:
            segments.append({"start_s": round(start, 6), "end_s": round(hi, 6)})
    return segments


# =========================================================================== #
# Hierarchical multi-channel chopping (coarsest channel that fits; else descend).
# =========================================================================== #
def channels_by_granularity(data: EpisodeData) -> list[dict[str, Any]]:
    """Per-channel reconciled interval spans, ordered COARSEST -> FINEST.

    Returns ``[{name, kind, spans:[{start_s,end_s,text}]}]`` sorted by descending
    mean span duration (the coarsest = whole-take narration first, finest = atomic
    actions last). Only spans with non-empty text and a real [start,end] are kept.
    """
    channels: list[dict[str, Any]] = []
    for ch in reconciled_annotations(data):
        spans = []
        for s in ch["segments"]:
            text = s.get("text")
            a, b = s.get("start_s"), s.get("end_s")
            if text is None or str(text).strip() == "":
                continue
            if a is None or b is None or b <= a:
                continue
            spans.append({"start_s": float(a), "end_s": float(b), "text": str(text)})
        if not spans:
            continue
        spans.sort(key=lambda s: s["start_s"])
        mean_dur = sum(s["end_s"] - s["start_s"] for s in spans) / len(spans)
        channels.append({"name": ch["name"], "kind": ch["kind"], "spans": spans, "mean_dur": mean_dur})
    channels.sort(key=lambda c: c["mean_dur"], reverse=True)  # coarsest first
    return channels


def chop_by_channels(
    channels: list[dict[str, Any]],
    *,
    max_clip_s: float,
    drop_shorter_than_s: float = 1.0,
    duration_s: float | None = None,
    _range: tuple[float, float] | None = None,
    _level: int = 0,
) -> list[dict[str, Any]]:
    """Chop using the coarsest annotation channel whose spans fit, else descend.

    Algorithm (per the spec): consider the channels coarsest->finest. Within the
    current time ``_range`` (whole episode at the top level), walk the coarsest
    channel's spans that overlap the range:
      * a span that fits (<= ``max_clip_s``) becomes a CLIP, tagged with that
        channel's name + the span's text;
      * a span TOO LONG is recursively re-chopped using the NEXT-finer channels
        restricted to that span's time range; if no finer channel exists, the span
        is hard-cut into <= ``max_clip_s`` pieces (carrying the coarse text).
    Each returned segment is ``{start_s, end_s, channel, text}``. Segments shorter
    than ``drop_shorter_than_s`` are dropped.
    """
    if not channels:
        return []
    rng_lo, rng_hi = _range if _range is not None else (0.0, float(duration_s) if duration_s else math.inf)
    coarse = channels[0]
    finer = channels[1:]
    segs: list[dict[str, Any]] = []
    for s in coarse["spans"]:
        lo, hi = max(s["start_s"], rng_lo), min(s["end_s"], rng_hi)
        if hi - lo < drop_shorter_than_s:
            continue
        length = hi - lo
        if length <= max_clip_s + 1e-6:
            segs.append({"start_s": round(lo, 6), "end_s": round(hi, 6),
                         "channel": coarse["name"], "text": s["text"]})
        elif finer:
            # descend: re-chop this too-long span with finer channels
            sub = chop_by_channels(finer, max_clip_s=max_clip_s, drop_shorter_than_s=drop_shorter_than_s,
                                   duration_s=duration_s, _range=(lo, hi), _level=_level + 1)
            if sub:
                segs.extend(sub)
            else:
                # finer channels had no coverage here -> hard-cut with coarse text
                segs.extend(_hard_cut(lo, hi, max_clip_s, drop_shorter_than_s, coarse["name"], s["text"]))
        else:
            # finest channel, still too long -> hard-cut, carry the (finest) text
            segs.extend(_hard_cut(lo, hi, max_clip_s, drop_shorter_than_s, coarse["name"], s["text"]))
    segs.sort(key=lambda r: r["start_s"])
    return segs


def _hard_cut(lo: float, hi: float, max_clip_s: float, drop_shorter_than_s: float,
              channel: str, text: str) -> list[dict[str, Any]]:
    out = []
    start = lo
    while hi - start > max_clip_s + 1e-6:
        out.append({"start_s": round(start, 6), "end_s": round(start + max_clip_s, 6),
                    "channel": channel, "text": text})
        start += max_clip_s
    if hi - start >= drop_shorter_than_s:
        out.append({"start_s": round(start, 6), "end_s": round(hi, 6), "channel": channel, "text": text})
    return out


def _numbered_text(texts: list[str]) -> str:
    """Combine multiple annotation texts into a single numbered-list label.

    One text -> returned unchanged; many -> "1) A 2) B 3) C". Empty/blank texts are
    skipped; consecutive duplicates are collapsed (a coalesced run of the same action
    reads as one item rather than "1) walk 2) walk").
    """
    clean: list[str] = []
    for t in texts:
        s = str(t).strip() if t is not None else ""
        if not s:
            continue
        if clean and clean[-1] == s:
            continue
        clean.append(s)
    if not clean:
        return ""
    if len(clean) == 1:
        return clean[0]
    return " ".join(f"{i + 1}) {t}" for i, t in enumerate(clean))


def coalesce_short_segments(
    segments: list[dict[str, Any]],
    *,
    min_duration_s: float,
    max_clip_s: float,
) -> list[dict[str, Any]]:
    """Coalesce too-short clips with the following clip(s) to reach ``min_duration_s``.

    When a segment's duration ``(end_s - start_s)`` is below ``min_duration_s`` we
    extend it to absorb the NEXT segment(s) in time order, combining their texts into
    a numbered list (:func:`_numbered_text`). Coalescing stops once the merged clip
    reaches ``min_duration_s`` or absorbing the next would exceed ``max_clip_s`` (the
    hard ceiling always wins). A trailing short segment with nothing left to merge is
    kept as-is. ``min_duration_s <= 0`` disables this (returns ``segments`` unchanged).

    The merged segment keeps the FIRST segment's driving ``channel``; its ``text`` is
    the numbered concatenation of every absorbed segment's text. Operates on the
    already-chopped, video-clock segment list; assumes ``segments`` is sorted by start.
    """
    if min_duration_s <= 0 or not segments:
        return segments
    segs = sorted(segments, key=lambda s: s["start_s"])
    out: list[dict[str, Any]] = []
    i = 0
    n = len(segs)
    while i < n:
        cur = dict(segs[i])
        texts = [cur.get("text", "")]
        j = i + 1
        # Absorb following segments while we're still short AND the merge fits.
        while (cur["end_s"] - cur["start_s"]) < min_duration_s - 1e-6 and j < n:
            nxt = segs[j]
            if (nxt["end_s"] - cur["start_s"]) > max_clip_s + 1e-6:
                break  # absorbing would blow past the max-clip ceiling
            cur["end_s"] = nxt["end_s"]
            texts.append(nxt.get("text", ""))
            j += 1
        cur["text"] = _numbered_text(texts)
        if j > i + 1:
            cur["coalesced"] = j - i  # how many source segments merged (provenance)
        cur["start_s"] = round(cur["start_s"], 6)
        cur["end_s"] = round(cur["end_s"], 6)
        out.append(cur)
        i = j if j > i else i + 1
    return out


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


def _probe_nframes(path: Path) -> int:
    """Exact decoded frame count of an mp4 (so frame<->gaze-point mapping is 1:1).

    Counts packets on the video stream (accurate for the short, freshly-encoded clips
    we emit; the index read is cheap). Returns 0 on failure.
    """
    r = _run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
              "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)])
    try:
        return int((r.stdout or "0").strip().split(",")[0])
    except (ValueError, IndexError):
        return 0


def resample_segment_from_local(
    full_src: Path,
    out_path: Path,
    *,
    start_s: float,
    seg_len: float,
    fps: float,
    side: int,
) -> dict[str, Any]:
    """Trim+resample one segment from an ALREADY-LOCAL full mp4 (no pull, no delete).

    Single ffmpeg pass: seek to ``start_s``, take ``seg_len`` seconds, resample to
    ``side``x``side`` aspect-preserving pad at ``fps``, h264. Used by the molmoact2
    path which pulls the big source ONCE and trims every segment from it (vs the
    per-segment re-pull the legacy window path did). Returns the EXACT encoded frame
    count (``nb_frames``) so the gaze emitter can place one point per real video frame.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vf = (
        f"fps={fps},"
        f"scale={side}:{side}:force_original_aspect_ratio=decrease,"
        f"pad={side}:{side}:(ow-iw)/2:(oh-ih)/2"
    )
    cmd = [
        "ffmpeg", "-y", "-ss", f"{start_s:.3f}", "-i", str(full_src), "-t", f"{seg_len:.3f}",
        "-vf", vf, "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path),
    ]
    r = _run(cmd)
    if r.returncode != 0 or not out_path.exists():
        raise RuntimeError(f"segment resample failed: {r.stderr[-600:]}")
    meta = ov._probe(out_path)
    return {"path": str(out_path), "width": meta["width"], "height": meta["height"],
            "fps": fps, "duration_s": meta.get("duration"), "start_s": start_s,
            "nb_frames": _probe_nframes(out_path)}


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
    episode_lists: dict[str, list[str]] | None = None,
    sample_extra: dict[str, dict[str, Any]] | None = None,
    output_format: str = "molmoact2",
    profile: str = DEFAULT_PROFILE,
    fps: float = DEFAULT_FPS,
    num_frames: int = DEFAULT_NUM_FRAMES,
    stride: int = DEFAULT_STRIDE,
    resolution: int = DEFAULT_RESOLUTION,
    temporality: str = DEFAULT_TEMPORALITY,
    window_s: float | None = 30.0,
    # MolmoAct2 / chopping knobs
    max_clip_s: float = 20.0,
    merge_gap_s: float = 1.0,
    drop_shorter_than_s: float = 1.0,
    min_duration_s: float = 0.0,
    max_frames: int = ma.DEFAULT_MAX_FRAMES,
    prompt: str = ma.DEFAULT_PROMPT,
    interesting_maps: dict[str, dict[str, Any]] | None = None,
    workers: int | None = None,
    puller: Puller | None = None,
    reuse_bundle: bool = True,
) -> dict[str, Any]:
    """Build a training manifest.

    ``output_format`` selects the emitter:
      * ``molmoact2`` (default): annotation-bounded clip SEGMENTS, each a
        ``resolution``x``resolution`` @ ``fps`` mp4, with per-frame pixel gaze points
        in the Molmo2VideoPoint shape (variable length padded to ``max_frames``).
      * ``qwen``: the legacy fixed sliding-window single-point profile.

    ``episode_lists`` maps slug -> list of episode ids (multiple per dataset). Falls
    back to ``episodes`` (one per dataset) then ``recipes/_sample_episodes.json``.
    ``dataset_filters`` (cull / exclude globs / gaze gap) are honored. For nymeria
    (or any dataset) an ``interesting_maps[slug]`` filter restricts segments to
    classified-interesting regions.
    """
    from . import curate_readers as cr

    out_root = Path(out_root)
    (out_root / "videos").mkdir(parents=True, exist_ok=True)
    puller = puller or Puller()
    interesting_maps = interesting_maps or {}

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

    # Build the (slug -> [episode_ids]) work list, honoring cull + exclude globs.
    def episodes_for(slug: str) -> list[str]:
        if episode_lists and slug in episode_lists:
            ids = list(episode_lists[slug])
        else:
            ids = [sample_map[slug]["episode_id"]]
        return cr.filter_episode_ids(slug, ids)

    if output_format == "molmoact2":
        import concurrent.futures
        import os

        # Flat work list of (slug, episode_id), honoring cull + exclude globs.
        jobs: list[tuple[str, str]] = []
        for slug in slugs:
            if cr.is_culled(slug):
                rows_report.append({"dataset": slug, "culled": True})
                continue
            for episode_id in episodes_for(slug):
                jobs.append((slug, episode_id))

        n_workers = max(1, workers if workers else (os.cpu_count() or 4))

        # Per-episode positional tokens (derived from each episode id) must NOT be
        # inherited from the sample episode when iterating a multi-episode list -- else
        # e.g. egtea's sample session leaks onto every other episode's video path.
        per_episode_keys = {"session", "participant", "take_name", "take_uid", "stem", "video_uid"}
        multi = bool(episode_lists)

        def run_job(job: tuple[str, str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
            slug, episode_id = job
            extra = {
                k: v for k, v in sample_map[slug].items()
                if k != "episode_id" and not (multi and k in per_episode_keys)
            }
            # Each episode gets its OWN Puller workdir so parallel pulls/trims never collide.
            ep_puller = Puller(
                ssh_host=puller.ssh_host, remote_root=puller.remote_root,
                local_root=puller.local_root,
                workdir=out_root / "_work" / _safe(f"{slug}__{episode_id}"),
            )
            import sys as _sys
            print(f"[curate] start {slug}:{episode_id}", file=_sys.stderr, flush=True)
            res = _build_molmoact_episode(
                slug, episode_id, extra, out_root, ep_puller,
                fps=fps, resolution=resolution, max_frames=max_frames,
                max_clip_s=max_clip_s, merge_gap_s=merge_gap_s, drop_shorter_than_s=drop_shorter_than_s,
                min_duration_s=min_duration_s,
                prompt=prompt, reuse_bundle=reuse_bundle,
                interesting=interesting_maps.get(slug),
            )
            print(f"[curate] done  {slug}:{episode_id} clips={len(res[0])}", file=_sys.stderr, flush=True)
            return res

        if n_workers == 1 or len(jobs) <= 1:
            results = [run_job(j) for j in jobs]
        else:
            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(n_workers, len(jobs))) as ex:
                futs = {ex.submit(run_job, j): j for j in jobs}
                for fut in concurrent.futures.as_completed(futs):
                    results.append(fut.result())
        for examples, rep in results:
            all_examples.extend(examples)
            rows_report.append(rep)

        return _write_manifest_outputs(
            out_root, all_examples, rows_report,
            schema=ma.MOLMOACT_SCHEMA,
            report_meta={
                "output_format": "molmoact2", "fps": fps, "resolution": resolution,
                "gaze_hz": fps, "points_per_video_frame": 1,
                "max_frames": max_frames or "unlimited", "max_clip_s": max_clip_s,
                "merge_gap_s": merge_gap_s, "drop_shorter_than_s": drop_shorter_than_s,
                "min_duration_s": min_duration_s, "workers": n_workers,
            },
        )

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


# =========================================================================== #
# MolmoAct2 per-episode builder (annotation-bounded segments -> video-point rows).
# =========================================================================== #
def _intersect_interesting(spans: list[dict[str, Any]], interesting: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Keep only spans overlapping a classified-interesting region.

    ``interesting`` is ``{"regions": [{"start_s","end_s","interesting":bool}, ...]}``
    on the video clock (e.g. the nymeria filter map for one take). Spans not
    overlapping any interesting region are dropped. If ``interesting`` is None the
    spans pass through unchanged.
    """
    if not interesting:
        return spans
    regions = [(r["start_s"], r["end_s"]) for r in interesting.get("regions", [])
               if r.get("interesting") and r.get("start_s") is not None and r.get("end_s") is not None]
    if not regions:
        return []
    kept = []
    for s in spans:
        a = s.get("start_s") if s.get("start_s") is not None else s.get("point_s")
        b = s.get("end_s") if s.get("end_s") is not None else s.get("point_s")
        if a is None:
            continue
        b = b if b is not None else a
        if any(a < rb and b > ra for ra, rb in regions):
            kept.append(s)
    return kept


def _filter_channels_interesting(channels: list[dict[str, Any]], interesting: dict[str, Any]) -> list[dict[str, Any]]:
    """Drop each channel's spans that don't overlap an interesting region.

    Applies the same overlap test as ``_intersect_interesting`` per channel, so the
    hierarchical chopper only ever sees interesting spans (at every granularity).
    Channels left with no spans are removed.
    """
    regions = [(r["start_s"], r["end_s"]) for r in interesting.get("regions", [])
               if r.get("interesting") and r.get("start_s") is not None and r.get("end_s") is not None]
    if not regions:
        return []
    out = []
    for ch in channels:
        kept = [s for s in ch["spans"]
                if any(s["start_s"] < rb and s["end_s"] > ra for ra, rb in regions)]
        if kept:
            out.append({**ch, "spans": kept})
    return out


def _build_molmoact_episode(
    slug: str,
    episode_id: str,
    extra: dict[str, Any],
    out_root: Path,
    puller: Puller,
    *,
    fps: float,
    resolution: int,
    max_frames: int,
    max_clip_s: float,
    merge_gap_s: float,
    drop_shorter_than_s: float,
    min_duration_s: float,
    prompt: str,
    reuse_bundle: bool,
    interesting: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build MolmoAct2 video-point rows for one episode: filter -> chop -> per-seg clip."""
    from . import curate_readers as cr

    try:
        data = build_episode_data(slug, episode_id, puller, sample_extra=extra, reuse=reuse_bundle)
    except Exception as exc:  # noqa
        return [], {"dataset": slug, "episode": episode_id, "error": str(exc)}

    # Gaze validity-gap cull (egome 0.25, via dataset_filters).
    gap_thr = cr.dataset_filters(slug).get("max_gaze_valid_gap_s")
    if gap_thr is not None:
        passes, observed = check_gaze_validity_gaps(data, float(gap_thr))
        if not passes:
            return [], {"dataset": slug, "episode": episode_id,
                        "culled_gaze_gap": True, "max_gap_s": round(observed, 3)}

    src_w = int(data.video.get("width") or resolution)
    src_h = int(data.video.get("height") or resolution)
    duration_s = float(data.video.get("duration_s") or 0.0)
    pad = PadTransform(src_w=src_w, src_h=src_h, side=resolution)

    # Per-channel spans ordered coarsest -> finest, optionally restricted to the
    # episode's classified-interesting regions (nymeria etc.). `interesting` is the
    # dataset's whole map {episode_id: {regions:[...]}}.
    channels = channels_by_granularity(data)
    ep_interesting = None
    if interesting:
        ep_interesting = interesting.get(episode_id) if episode_id in interesting else (
            interesting if "regions" in interesting else None
        )
    if ep_interesting:
        channels = _filter_channels_interesting(channels, ep_interesting)

    # Optional Molmo fixed-length cap: max_frames>0 means no clip may exceed
    # max_frames/fps seconds (keeps the 1:1 frame<->point map within the cap). The
    # tighter of (user max_clip_s, frame cap) wins.
    eff_max_clip_s = max_clip_s
    if max_frames and max_frames > 0:
        eff_max_clip_s = min(max_clip_s, max_frames / fps)

    # Hierarchical chop: coarsest channel that fits <= max_clip; else descend to finer.
    # Each segment carries the channel + text of the span that drove its boundaries.
    segments = chop_by_channels(
        channels, max_clip_s=eff_max_clip_s, drop_shorter_than_s=drop_shorter_than_s, duration_s=duration_s or None,
    )
    # Coalesce too-short clips with following clip(s) up to min_duration_s (text ->
    # numbered list). Disabled when min_duration_s <= 0.
    n_before_coalesce = len(segments)
    segments = coalesce_short_segments(
        segments, min_duration_s=min_duration_s, max_clip_s=eff_max_clip_s,
    )
    if not segments:
        return [], {"dataset": slug, "episode": episode_id, "segments": 0,
                    "note": "no annotation-bounded segments"}

    # Project the gaze track once (source px, video clock). Subsample raw samples to
    # ~2.5x the grid fps before projecting (Aria CPF projection is ~60s for 30k
    # samples; we only resample to `fps` downstream, so projecting all is wasteful).
    times, xs, ys, inf = projected_gaze_track(data, puller, project_max_hz=max(8.0, 2.0 * fps))
    # Resample to the canonical fps grid over the whole episode (video clock).
    grid_dur = duration_s or (segments[-1]["end_s"] if segments else 0.0)
    resampled = resample_track_linear(
        times, xs, ys, inf, fps=fps, duration_s=grid_dur, max_gap_s=1.0,
    )

    # Pull the full source mp4 ONCE, trim every segment from it locally, delete at end.
    # (The legacy window path re-pulled per segment -> 25x downloads of a 369MB take.)
    try:
        full_src = puller.pull(data.video["path"])
    except Exception as exc:  # noqa
        return [], {"dataset": slug, "episode": episode_id, "error": f"pull: {exc}"}

    examples: list[dict[str, Any]] = []
    for k, seg in enumerate(segments):
        seg_start, seg_end = seg["start_s"], seg["end_s"]
        seg_len = seg_end - seg_start
        video_rel = f"videos/{slug}/{_safe(episode_id)}__seg{k}.mp4"
        try:
            vmeta = resample_segment_from_local(
                full_src, out_root / video_rel,
                start_s=seg_start, seg_len=seg_len, fps=fps, side=resolution,
            )
        except Exception as exc:  # noqa
            examples.append({"dataset": slug, "episode_id": episode_id, "seg_index": k,
                             "error": f"video: {exc}"})
            continue
        # gaze_hz == video_fps: one gaze point per REAL video frame. Use the EXACT
        # encoded frame count so points[j] <-> video frame j is a true 1:1 index map.
        n_frames = vmeta.get("nb_frames") or int(round(seg_len * fps))
        frames, num_real = ma.sample_segment_frames(
            resampled.times_s, resampled.px, resampled.py, resampled.in_frame, resampled.valid,
            seg_start_s=seg_start, seg_end_s=seg_end, fps=fps, n_frames=n_frames, pad=pad,
        )
        if num_real == 0:
            continue
        # Each segment carries the channel + text of the span that drove its boundaries.
        examples.append(ma.build_molmoact_row(
            dataset=slug, episode_id=episode_id, seg_index=k, video_rel=video_rel,
            seg_start_s=seg_start, seg_end_s=seg_end, frames=frames, num_real=num_real,
            fps=fps, side=resolution,
            annotation_text=seg.get("text"), annotation_channel=seg.get("channel"),
            prompt=prompt,
        ))

    # Delete the big pulled source (keep small ones like egome/egtea harmlessly).
    # NEVER delete an in-place local_root/nfs source -- only our own scp'd temp copy.
    try:
        if puller.owns(full_src) and full_src.exists() and full_src.stat().st_size > 5_000_000:
            full_src.unlink()
    except OSError:
        pass

    real = [e for e in examples if "error" not in e]
    n_coalesced = sum(1 for s in segments if s.get("coalesced"))
    frame_counts = [e.get("num_frames", 0) for e in real]
    rep = {
        "dataset": slug, "episode": episode_id,
        "segments": len(segments), "clips": len(real),
        "segments_before_coalesce": n_before_coalesce, "coalesced_segments": n_coalesced,
        "frames_min": min(frame_counts) if frame_counts else 0,
        "frames_max": max(frame_counts) if frame_counts else 0,
        "frames_total": sum(frame_counts),
        "source_dims": f"{src_w}x{src_h}", "duration_s": round(duration_s, 3),
        "projection": data.projection_method, "gaze_space": data.gaze_space,
        "interesting_filtered": interesting is not None,
    }
    return examples, rep


def _write_manifest_outputs(
    out_root: Path,
    examples: list[dict[str, Any]],
    rows_report: list[dict[str, Any]],
    *,
    schema: dict[str, Any],
    report_meta: dict[str, Any],
) -> dict[str, Any]:
    _write_jsonl(examples, out_root / "manifest.jsonl")
    _write_table(examples, out_root / "manifest.parquet")
    (out_root / "schema.json").write_text(json.dumps(schema, indent=2, sort_keys=True), encoding="utf-8")
    report = {**report_meta, "total_examples": len(examples), "episodes": rows_report}
    (out_root / "training_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
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
    # table form and keep the full nested record in manifest.jsonl. Handle both the
    # qwen single-point rows and the molmoact2 video-point rows.
    flat = []
    for r in rows:
        if "error" in r:
            continue
        if "target" in r:  # qwen single-point row
            flat.append({
                "id": r["id"], "dataset": r["dataset"], "episode_id": r["episode_id"],
                "anchor_time_s": r.get("anchor_time_s"), "num_frames": r.get("num_frames"),
                "fps": r.get("fps"),
                "x_norm": r["target"][0], "y_norm": r["target"][1],
                "x_1000": r["target_1000"][0], "y_1000": r["target_1000"][1],
                "target_valid": r.get("target_valid"), "annotation_text": r.get("annotation_text"),
            })
        else:  # molmoact2 video-point row
            md = r.get("metadata", {})
            flat.append({
                "id": r["id"], "dataset": r["dataset"], "episode_id": r["episode_id"],
                "seg_index": r.get("seg_index"), "video": r.get("video"),
                "num_frames": r.get("num_frames"), "num_frames_real": r.get("num_frames_real"),
                "fps": r.get("fps"), "resolution": r.get("resolution"),
                "clip_start_time": md.get("clip_start_time"), "clip_end_time": md.get("clip_end_time"),
                "annotation_text": md.get("annotation_text"),
            })
    write_table(flat, path)
