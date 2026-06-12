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
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import molmo2 as ma
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
# so molmo2.py can reuse them without a circular import. Re-exported here for
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
# A point annotation (no duration, e.g. ego-exo4d atomic_descriptions) is widened to a
# small interval centred on the event so it can DRIVE a clip; the clip then grows to
# min_duration_s via coalescing. Half-width on each side of the point.
POINT_WINDOW_S = 1.0


def channels_by_granularity(data: EpisodeData, *, point_window_s: float = POINT_WINDOW_S) -> list[dict[str, Any]]:
    """Per-channel reconciled spans, ordered COARSEST -> FINEST, for chop DRIVING.

    Returns ``[{name, kind, role, spans:[{start_s,end_s,text}]}]`` sorted by descending
    mean span duration (coarsest first). Only spans with non-empty text are kept.

    Channel ``clip_role`` (recipe-driven):
      * ``driver`` (default): drives clip boundaries/labels.
      * ``context``: NOT returned here (never drives), but still attached to clips as
        AUXILIARY annotation (see ``auxiliary_channel_spans``). e.g. nymeria
        activity_summary -- its bundled text would leak conversation if it drove clips.
      * ``disabled``: excluded entirely (neither driver nor auxiliary). e.g. ego-exo4d
        expert commentary, which we do not want in the dataset at all.

    POINT channels (``kind == "point"``, e.g. ego-exo4d atomic_descriptions) carry no
    duration; each point is widened to ``[point - point_window_s, point + point_window_s]``
    so it can drive a (short) clip that coalescing then grows to min_duration.
    """
    roles = _channel_roles(data)
    channels: list[dict[str, Any]] = []
    for ch in reconciled_annotations(data):
        role = roles.get(ch["name"], "driver")
        if role in ("context", "disabled"):
            continue  # context = auxiliary only (added later); disabled = dropped
        spans = _channel_spans(ch, point_window_s)
        if not spans:
            continue
        mean_dur = sum(s["end_s"] - s["start_s"] for s in spans) / len(spans)
        channels.append({"name": ch["name"], "kind": ch["kind"], "role": role,
                         "spans": spans, "mean_dur": mean_dur})
    channels.sort(key=lambda c: c["mean_dur"], reverse=True)  # coarsest first
    return channels


def _channel_spans(ch: dict[str, Any], point_window_s: float) -> list[dict[str, Any]]:
    """Reconciled segments -> [{start_s,end_s,text}] intervals (points widened)."""
    spans = []
    for s in ch["segments"]:
        text = s.get("text")
        if text is None or str(text).strip() == "":
            continue
        a, b, p = s.get("start_s"), s.get("end_s"), s.get("point_s")
        if a is not None and b is not None and b > a:
            lo, hi = float(a), float(b)
        elif p is not None:                       # point event -> widen to a window
            lo, hi = float(p) - point_window_s, float(p) + point_window_s
        elif a is not None and (b is None or b <= a):  # open/zero-length start -> window
            lo, hi = float(a), float(a) + 2 * point_window_s
        else:
            continue
        if hi <= lo:
            continue
        spans.append({"start_s": lo, "end_s": hi, "text": str(text)})
    spans.sort(key=lambda s: s["start_s"])
    return spans


def _channel_roles(data: EpisodeData) -> dict[str, str]:
    """Map recipe annotation channel name -> clip_role (default 'driver')."""
    return {ch.get("name"): ch.get("clip_role", "driver")
            for ch in (data.recipe.get("annotations") or [])}


def _context_channels(data: EpisodeData) -> set[str]:
    """Names of channels marked ``clip_role: context`` (auxiliary only, never drive)."""
    return {n for n, r in _channel_roles(data).items() if r == "context"}


def auxiliary_channel_spans(data: EpisodeData, *, point_window_s: float = POINT_WINDOW_S) -> list[dict[str, Any]]:
    """All annotation spans usable as AUXILIARY context for a clip (item 4).

    Returns a flat list ``[{start_s,end_s,channel,text}]`` from every channel whose
    role is ``driver`` OR ``context`` (NOT ``disabled``). Points widened like the
    driver path. Used to attach every temporally-containing channel to each clip.
    """
    roles = _channel_roles(data)
    out: list[dict[str, Any]] = []
    for ch in reconciled_annotations(data):
        if roles.get(ch["name"], "driver") == "disabled":
            continue
        for s in _channel_spans(ch, point_window_s):
            out.append({"start_s": s["start_s"], "end_s": s["end_s"],
                        "channel": ch["name"], "text": s["text"]})
    return out


def annotations_covering_clip(
    aux_spans: list[dict[str, Any]],
    seg_start_s: float,
    seg_end_s: float,
    *,
    driver_channel: str | None,
    driver_text: str | None,
    min_overlap_frac: float = 0.5,
) -> list[dict[str, Any]]:
    """All annotation spans temporally CONTAINED WITHIN / overlapping a clip (item 4).

    Returns ``[{channel, text, start_s, end_s, overlap_s, source_duration_s}]`` for every
    auxiliary span that overlaps ``[seg_start_s, seg_end_s]`` by at least
    ``min_overlap_frac`` of the SPAN's own length (so a span the clip barely clips at the
    edge isn't attached). The span that DROVE the clip (matched by channel+text) is
    excluded here -- it is the clip's default annotation; the rest are auxiliary.

    ``start_s``/``end_s`` are clip-relative and CLAMPED to the clip window; ``overlap_s``
    is how much of the span lies inside the clip; ``source_duration_s`` is the FULL
    duration of the original (un-clamped) source annotation span on the video clock, so
    a consumer can tell how much of the underlying annotation the clip actually captures.
    """
    out: list[dict[str, Any]] = []
    seg_len = seg_end_s - seg_start_s
    for s in aux_spans:
        a, b = s["start_s"], s["end_s"]
        ov = min(b, seg_end_s) - max(a, seg_start_s)
        if ov <= 0:
            continue
        span_len = b - a
        if span_len > 0 and (ov / span_len) < min_overlap_frac and ov < seg_len - 1e-6:
            continue  # only a sliver of the span overlaps and it doesn't fill the clip
        # Skip the exact span that drove this clip (it's the default annotation).
        if (s["channel"] == driver_channel and driver_text is not None
                and str(s["text"]) in str(driver_text)):
            continue
        out.append({
            "channel": s["channel"],
            "text": s["text"],
            "start_s": round(max(0.0, a - seg_start_s), 3),
            "end_s": round(min(seg_len, b - seg_start_s), 3),
            "overlap_s": round(ov, 3),
            "source_duration_s": round(span_len, 3),
        })
    out.sort(key=lambda r: (r["channel"], r["start_s"]))
    return out


def build_final_annotation(
    bundle: dict[str, Any] | None,
    *,
    source_channel: str | None,
    source_text: str | None,
    auxiliary: list[dict[str, Any]],
) -> str | None:
    """Assemble the dataset's FINAL training annotation from the shipped channels.

    Driven by the recipe ``annotation_bundle`` block (item: per-dataset bundling). All
    channels stay on the clip (source = default, auxiliary = the rest); this only adds
    a marked ``final_annotation`` string built per the policy:

      * ``source_only`` (ego-exo4d / egtea / hd-epic / nymeria): final == source text.
      * ``holoassist_coarse_fine``: if the SOURCE channel is a fine action channel, the
        final is ``"<context: COARSE> <action: FINE>"`` where COARSE is the covering
        coarse-channel auxiliary text (or '' if none); if the SOURCE is the coarse
        channel, the final is just ``"<COARSE>"`` (== source text). Other source
        channels fall back to source text.

    ``bundle`` is the recipe's ``annotation_bundle`` dict; None / missing -> source text.
    """
    if not source_text:
        return source_text
    policy = (bundle or {}).get("policy", "source_only")
    if policy == "source_only" or not bundle:
        return source_text
    if policy == "holoassist_coarse_fine":
        coarse_ch = bundle.get("coarse_channel")
        fine_chs = set(bundle.get("fine_channels") or [])
        if source_channel in fine_chs:
            # pick the coarse auxiliary covering this clip (longest overlap if several)
            coarse_aux = [a for a in auxiliary if a.get("channel") == coarse_ch and a.get("text")]
            coarse_text = ""
            if coarse_aux:
                coarse_text = max(coarse_aux, key=lambda a: a.get("overlap_s", 0)).get("text", "")
            return f"<context: {coarse_text}> <action: {source_text}>"
        if source_channel == coarse_ch:
            return f"<{source_text}>"
        return source_text
    return source_text


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
    drop_unmergeable: bool = True,
) -> list[dict[str, Any]]:
    """Coalesce too-short clips with the following clip(s) to reach ``min_duration_s``.

    When a segment's duration ``(end_s - start_s)`` is below ``min_duration_s`` we
    extend it to absorb the NEXT segment(s) in time order, combining their texts into
    a numbered list (:func:`_numbered_text`). Coalescing stops once the merged clip
    reaches ``min_duration_s`` or absorbing the next would exceed ``max_clip_s`` (the
    hard ceiling always wins). ``min_duration_s <= 0`` disables this (returns
    ``segments`` unchanged).

    DROP RULE (item 3): a merged clip that STILL falls short of ``min_duration_s``
    (e.g. a single isolated egtea clip with no following segment to absorb, or a
    trailing remainder) is DROPPED when ``drop_unmergeable`` is True (the default) --
    we do not ship clips below the requested minimum duration. Set False to keep them.

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
        # Item 3: never ship a clip below the requested minimum duration.
        if drop_unmergeable and (cur["end_s"] - cur["start_s"]) < min_duration_s - 1e-6:
            i = j if j > i else i + 1
            continue
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
    reuse_existing: bool = False,
) -> dict[str, Any]:
    """Trim+resample one segment from an ALREADY-LOCAL full mp4 (no pull, no delete).

    Single ffmpeg pass: seek to ``start_s``, take ``seg_len`` seconds, resample to
    ``side``x``side`` aspect-preserving pad at ``fps``, h264. Used by the molmo2
    path which pulls the big source ONCE and trims every segment from it (vs the
    per-segment re-pull the legacy window path did). Returns the EXACT encoded frame
    count (``nb_frames``) so the gaze emitter can place one point per real video frame.

    When ``reuse_existing`` is True and ``out_path`` already exists as a valid
    ``side``x``side`` clip with >0 frames, the ffmpeg encode is SKIPPED and the existing
    clip is probed instead. This makes a manifest-recovery re-run (clips already on disk
    from a crashed run) reuse the encoded clips and only redo the cheap gaze/annotation
    assembly. ``full_src`` may be None in that case (not needed when reusing).
    """
    if reuse_existing and out_path.exists():
        meta = ov._probe(out_path)
        nf = _probe_nframes(out_path)
        if nf > 0 and int(meta.get("width") or 0) == side and int(meta.get("height") or 0) == side:
            return {"path": str(out_path), "width": meta["width"], "height": meta["height"],
                    "fps": fps, "duration_s": meta.get("duration"), "start_s": start_s,
                    "nb_frames": nf, "reused": True}
        # else: invalid/partial clip -> fall through and re-encode
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
# Process-parallel molmo2 extraction: per-episode worker + sharded dispatch.
#
# Manifest generation is CPU-bound in pure Python (gaze projection/resampling, row
# assembly), so a thread pool is GIL-bound and pins one core. We parallelize across
# PROCESSES instead -- each worker handles one episode end to end and persists its
# result to a per-episode shard. The shard cache is both the crash-safe resume log
# AND the cross-worker coordination: a worker skips an episode whose shard exists, so
# any number of processes (or a relaunch after a crash) converge with zero rework.
# =========================================================================== #
@dataclass
class _EpisodeJobConfig:
    """Immutable per-run knobs passed to each episode worker (picklable)."""

    out_root: str
    fps: float
    resolution: int
    max_frames: int
    max_clip_s: float
    merge_gap_s: float
    drop_shorter_than_s: float
    min_duration_s: float
    prompt: str
    reuse_bundle: bool
    reuse_clips: bool
    ssh_host: str | None
    remote_root: str
    local_root: str | None
    multi: bool  # iterating a multi-episode list -> don't inherit sample positional tokens


# Positional tokens (derived per episode id) must NOT be inherited from the sample
# episode in multi-episode mode -- else e.g. egtea's sample session leaks onto every
# episode's video path.
_PER_EPISODE_TOKEN_KEYS = {"session", "participant", "take_name", "take_uid", "stem", "video_uid"}


def _shard_path(out_root: Path, slug: str, episode_id: str) -> Path:
    return out_root / "_shards" / f"{_safe(slug)}__{_safe(episode_id)}.json"


def _run_one_episode(
    slug: str, episode_id: str, sample_entry: dict[str, Any],
    interesting: dict[str, Any] | None, cfg: _EpisodeJobConfig,
) -> str:
    """Build ONE episode and persist its shard. Returns a short status string.

    Module-level + picklable so it runs under ProcessPoolExecutor. Idempotent: if the
    episode's shard already exists it is skipped (cross-process / resume coordination).
    Any failure is caught and written as an error shard -- one bad take never aborts
    the run. The shard write is atomic (tmp + rename) so a crash can't corrupt it.
    """
    import sys as _sys
    out_root = Path(cfg.out_root)
    sp = _shard_path(out_root, slug, episode_id)
    if sp.exists():
        return f"skip {slug}:{episode_id}"  # already done (resume / another worker)

    extra = {
        k: v for k, v in sample_entry.items()
        if k != "episode_id" and not (cfg.multi and k in _PER_EPISODE_TOKEN_KEYS)
    }
    ep_puller = Puller(
        ssh_host=cfg.ssh_host, remote_root=cfg.remote_root, local_root=cfg.local_root,
        workdir=out_root / "_work" / _safe(f"{slug}__{episode_id}"),
    )
    print(f"[curate] start {slug}:{episode_id}", file=_sys.stderr, flush=True)
    try:
        examples, rep = _build_molmo2_episode(
            slug, episode_id, extra, out_root, ep_puller,
            fps=cfg.fps, resolution=cfg.resolution, max_frames=cfg.max_frames,
            max_clip_s=cfg.max_clip_s, merge_gap_s=cfg.merge_gap_s,
            drop_shorter_than_s=cfg.drop_shorter_than_s, min_duration_s=cfg.min_duration_s,
            prompt=cfg.prompt, reuse_bundle=cfg.reuse_bundle,
            interesting=interesting, reuse_clips=cfg.reuse_clips,
        )
    except Exception as exc:  # noqa
        import traceback as _tb
        print(f"[curate] ERROR {slug}:{episode_id}: {exc}", file=_sys.stderr, flush=True)
        _tb.print_exc(file=_sys.stderr)
        examples, rep = [], {"dataset": slug, "episode": episode_id, "error": str(exc)}

    sp.parent.mkdir(parents=True, exist_ok=True)
    tmp = sp.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps({"examples": examples, "report": rep}, sort_keys=True), encoding="utf-8")
    tmp.replace(sp)  # atomic
    print(f"[curate] done  {slug}:{episode_id} clips={len(examples)}", file=_sys.stderr, flush=True)
    return f"done {slug}:{episode_id} clips={len(examples)}"


def _default_workers(n_jobs: int) -> int:
    """Process count: one per core, capped to the job count, leaving 2 cores free."""
    return max(1, min(n_jobs, (os.cpu_count() or 4) - 2))


def _build_molmo2_manifest(
    out_root: Path,
    slugs: list[str],
    sample_map: dict[str, dict[str, Any]],
    episodes_for,
    *,
    multi: bool,
    interesting_maps: dict[str, dict[str, Any]],
    puller: Puller,
    workers: int | None,
    fps: float, resolution: int, max_frames: int, max_clip_s: float,
    merge_gap_s: float, drop_shorter_than_s: float, min_duration_s: float,
    prompt: str, reuse_bundle: bool, reuse_clips: bool,
) -> dict[str, Any]:
    """Process-parallel molmo2 extraction with crash-safe per-episode shard caching.

    One worker PROCESS per episode (pool size = ``workers``, default ~all cores). Each
    worker is idempotent and self-coordinating via the shard cache, so this is fully
    resumable: rerun the same command after a crash and only un-sharded episodes run.
    Final manifest is the concatenation of all shards.
    """
    import concurrent.futures
    import sys as _sys
    from . import curate_readers as cr

    (out_root / "_shards").mkdir(parents=True, exist_ok=True)
    rows_report: list[dict[str, Any]] = []

    # Flat work list of (slug, episode_id), honoring cull + exclude globs.
    jobs: list[tuple[str, str]] = []
    for slug in slugs:
        if cr.is_culled(slug):
            rows_report.append({"dataset": slug, "culled": True})
            continue
        for episode_id in episodes_for(slug):
            jobs.append((slug, episode_id))

    pending = [j for j in jobs if not _shard_path(out_root, *j).exists()]
    n_cached = len(jobs) - len(pending)
    n_workers = max(1, workers) if workers else _default_workers(len(pending) or 1)
    print(f"[curate] molmo2: {len(jobs)} episodes ({n_cached} already done, {len(pending)} pending) "
          f"on {n_workers} worker process(es)", file=_sys.stderr, flush=True)

    cfg = _EpisodeJobConfig(
        out_root=str(out_root), fps=fps, resolution=resolution, max_frames=max_frames,
        max_clip_s=max_clip_s, merge_gap_s=merge_gap_s, drop_shorter_than_s=drop_shorter_than_s,
        min_duration_s=min_duration_s, prompt=prompt, reuse_bundle=reuse_bundle,
        reuse_clips=reuse_clips, ssh_host=puller.ssh_host, remote_root=puller.remote_root,
        local_root=str(puller.local_root) if puller.local_root else None, multi=multi,
    )

    # Dispatch pending episodes across PROCESSES (CPU-bound pure-Python work; threads
    # would be GIL-bound). 1 worker / 1 job -> run inline (simpler, debuggable).
    if pending:
        if n_workers == 1 or len(pending) == 1:
            for slug, ep in pending:
                _run_one_episode(slug, ep, sample_map[slug], interesting_maps.get(slug), cfg)
        else:
            ctx = __import__("multiprocessing").get_context("spawn")
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
                futs = [ex.submit(_run_one_episode, slug, ep, sample_map[slug],
                                  interesting_maps.get(slug), cfg) for slug, ep in pending]
                for fut in concurrent.futures.as_completed(futs):
                    try:
                        fut.result()  # shard already persisted in the worker
                    except Exception as exc:  # noqa: a worker crash shouldn't sink the pool
                        print(f"[curate] WARN worker crashed: {exc}", file=_sys.stderr, flush=True)

    # Assemble the manifest from ALL shards (cached + just-built).
    all_examples: list[dict[str, Any]] = []
    for slug, episode_id in jobs:
        sp = _shard_path(out_root, slug, episode_id)
        if not sp.exists():
            continue
        try:
            d = json.loads(sp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[curate] WARN bad shard {sp.name}: {exc}", file=_sys.stderr, flush=True)
            continue
        all_examples.extend(d.get("examples") or [])
        if d.get("report") is not None:
            rows_report.append(d["report"])

    return _write_manifest_outputs(
        out_root, all_examples, rows_report,
        schema=ma.MOLMO2_SCHEMA,
        report_meta={
            "output_format": "molmo2", "fps": fps, "resolution": resolution,
            "gaze_hz": fps, "points_per_video_frame": 1,
            "max_frames": max_frames or "unlimited", "max_clip_s": max_clip_s,
            "merge_gap_s": merge_gap_s, "drop_shorter_than_s": drop_shorter_than_s,
            "min_duration_s": min_duration_s, "workers": n_workers,
            "episodes_total": len(jobs), "episodes_cached_on_resume": n_cached,
        },
    )


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
    output_format: str = "molmo2",
    profile: str = DEFAULT_PROFILE,
    fps: float = DEFAULT_FPS,
    num_frames: int = DEFAULT_NUM_FRAMES,
    stride: int = DEFAULT_STRIDE,
    resolution: int = DEFAULT_RESOLUTION,
    temporality: str = DEFAULT_TEMPORALITY,
    window_s: float | None = 30.0,
    # Molmo2 / chopping knobs
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
    reuse_clips: bool = False,
) -> dict[str, Any]:
    """Build a training manifest.

    ``output_format`` selects the emitter:
      * ``molmo2`` (default): annotation-bounded clip SEGMENTS, each a
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

    if output_format == "molmo2":
        return _build_molmo2_manifest(
            out_root, slugs, sample_map, episodes_for, multi=bool(episode_lists),
            interesting_maps=interesting_maps, puller=puller, workers=workers,
            fps=fps, resolution=resolution, max_frames=max_frames, max_clip_s=max_clip_s,
            merge_gap_s=merge_gap_s, drop_shorter_than_s=drop_shorter_than_s,
            min_duration_s=min_duration_s, prompt=prompt, reuse_bundle=reuse_bundle,
            reuse_clips=reuse_clips,
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
# Molmo2 per-episode builder (annotation-bounded segments -> video-point rows).
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

    CHANNEL-AWARE gating (fixes the conversational leak): each annotation channel is
    filtered against the interesting regions FROM ITS OWN CHANNEL only. The classifier
    labels every channel's spans; pooling them let a coarse ``activity_summary`` region
    keep fine ``atomic_action`` idle/transitional sub-spans (standing, listening,
    gesticulating) that fall inside it -> conversation leaked into the clips. Matching
    channel-to-channel means a fine span survives only if the classifier marked a fine
    span (overlapping it) interesting.

    Back-compat: if the map's regions carry no ``channel`` (older maps) we fall back to
    the legacy any-channel overlap so existing maps still work.
    """
    all_regions = [r for r in interesting.get("regions", [])
                   if r.get("interesting") and r.get("start_s") is not None and r.get("end_s") is not None]
    if not all_regions:
        return []
    has_channel = any(r.get("channel") is not None for r in all_regions)

    out = []
    for ch in channels:
        if has_channel:
            regions = [(r["start_s"], r["end_s"]) for r in all_regions
                       if r.get("channel") == ch["name"]]
        else:
            regions = [(r["start_s"], r["end_s"]) for r in all_regions]
        kept = [s for s in ch["spans"]
                if any(s["start_s"] < rb and s["end_s"] > ra for ra, rb in regions)]
        if kept:
            out.append({**ch, "spans": kept})
    return out


def _build_molmo2_episode(
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
    reuse_clips: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build Molmo2 video-point rows for one episode: filter -> chop -> per-seg clip."""
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
    bundle_cfg = data.recipe.get("annotation_bundle")   # per-dataset final-annotation policy

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

    # All annotation spans (driver + context channels; disabled excluded) for attaching
    # every temporally-containing channel to each clip as auxiliary annotations (item 4).
    aux_spans = auxiliary_channel_spans(data)

    # Project the gaze track once (source px, video clock). Subsample raw samples to
    # ~2.5x the grid fps before projecting (Aria CPF projection is ~60s for 30k
    # samples; we only resample to `fps` downstream, so projecting all is wasteful).
    times, xs, ys, inf = projected_gaze_track(data, puller, project_max_hz=max(8.0, 2.0 * fps))
    # Resample to the canonical fps grid over the whole episode (video clock).
    grid_dur = duration_s or (segments[-1]["end_s"] if segments else 0.0)
    resampled = resample_track_linear(
        times, xs, ys, inf, fps=fps, duration_s=grid_dur, max_gap_s=1.0,
    )

    # In reuse_clips (manifest-recovery) mode, skip pulling the big source mp4 entirely
    # IF every segment clip already exists on disk; only pull when an encode is needed.
    all_clips_present = reuse_clips and all(
        (out_root / f"videos/{slug}/{_safe(episode_id)}__seg{k}.mp4").exists()
        for k in range(len(segments))
    )
    full_src = None
    if not all_clips_present:
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
                reuse_existing=reuse_clips,
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
        # Item 4: attach every other annotation channel temporally covering this clip
        # as AUXILIARY context (the driving span is the default annotation).
        aux = annotations_covering_clip(
            aux_spans, seg_start, seg_end,
            driver_channel=seg.get("channel"), driver_text=seg.get("text"),
        )
        # FINAL training annotation assembled per the recipe annotation_bundle policy
        # (all channels still shipped; this is an additional marked output).
        final_anno = build_final_annotation(
            bundle_cfg, source_channel=seg.get("channel"),
            source_text=seg.get("text"), auxiliary=aux,
        )
        # Each segment carries the channel + text of the span that drove its boundaries.
        examples.append(ma.build_molmo2_row(
            dataset=slug, episode_id=episode_id, seg_index=k, video_rel=video_rel,
            seg_start_s=seg_start, seg_end_s=seg_end, frames=frames, num_real=num_real,
            fps=fps, side=resolution,
            annotation_text=seg.get("text"), annotation_channel=seg.get("channel"),
            auxiliary_annotations=aux, final_annotation=final_anno,
            prompt=prompt,
        ))

    # Delete the big pulled source (keep small ones like egome/egtea harmlessly).
    # NEVER delete an in-place local_root/nfs source -- only our own scp'd temp copy.
    # full_src is None when reuse_clips skipped the pull entirely.
    try:
        if full_src is not None and puller.owns(full_src) and full_src.exists() and full_src.stat().st_size > 5_000_000:
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
    # qwen single-point rows and the molmo2 video-point rows.
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
        else:  # molmo2 video-point row
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
