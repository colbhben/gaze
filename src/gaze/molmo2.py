"""Molmo2 video-point finetuning manifest emitter.

Builds ``Molmo2VideoPoint``-style training rows (confirmed against
``allenai/molmo2`` ``olmo/data/molmo2_datasets.py``) for the gaze task:
predict where the camera wearer is looking, per frame, over a short video clip.

Confirmed Molmo2 video-point format (from source):
  * multi-frame video; the video-POINTING/track path defaults to 6 fps
    (``TrackingDataset.VIDEO_FPS = 6``; 2 fps is the general-video path, not ours).
    The only fps rule is integer divisibility -- ``sampling_fps`` must divide
    ``video_fps`` (trivially true at our 1:1 ``gaze_hz == video_fps``), capped at
    ``MAX_VIDEO_FPS = 10``. There is NO 0.5 s timestamp grid / divisibility assert
    (verified against allenai/molmo2 ``olmo/data/video_loader.py`` +
    ``*_video_track_datasets.py``, 2026-06-11);
  * per-frame frames preprocessed to 378x378 aspect-preserving (resize+pad);
  * points stored as RAW PIXEL ``{"x": float, "y": float}`` (one list per frame),
    NOT normalized and NOT <point> tokens (the model tokenizes internally);
  * ``timestamps`` clip-relative plain float seconds = frame ``j`` at ``j/fps``
    (no grid snapping);
  * example = ``message_list`` chat (user: text + {"type":"video"}; assistant: answer)
    with ``metadata{clip_start_time, clip_end_time}`` and ``style:"video_point"``.

Design decisions (user-confirmed):
  * per-frame gaze targets (gaze is dynamic);
  * canonical fps is user-defined; ALL datasets resample DOWN to it (sources are
    24-30 fps, always >= canonical);
  * gaze_hz == video_fps: we emit EXACTLY ONE gaze point per real video frame
    (1:1, variable length, NO cap and NO padding). The on-disk clip is encoded at
    the canonical fps and ``points[j]`` is the gaze at video frame ``j``. The
    training intent is to feed the first frame (t0) -- or first chunk -- of gaze and
    predict t0 gaze from the future frames + prompt, so a fixed ``max_frames`` /
    padding-to-max convention is intentionally PUNTED (see ``--max-frames`` below for
    an optional cap). Frames whose gaze is invalid/out-of-frame carry an empty point
    list (masked from loss) but still occupy their frame slot, so frame<->point
    alignment is never broken.

We keep the normalized ``[0,1]`` and integer ``0-1000`` forms per frame in a
``provenance`` sidecar (user: "match Molmo2 + keep normalized").
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Reuse the validated pad geometry from the contract base (no circular import).
from .curate import PadTransform, clamp01, to_bins


DEFAULT_PROMPT = "Point to where the camera wearer is looking."
DEFAULT_FPS = 2.0
# 0 / None => unlimited (one gaze point per video frame). A positive value caps the
# clip to that many frames (via an effective max-clip duration of max_frames/fps),
# preserving the 1:1 frame<->point mapping. Kept as an optional knob for Molmo's
# fixed-length conventions, which are otherwise punted.
DEFAULT_MAX_FRAMES = 0
DEFAULT_RESOLUTION = 378


@dataclass
class FrameGaze:
    """One sampled frame's gaze target on the padded `side`x`side` frame."""

    t_s: float            # clip-relative time (s), on the 1/fps grid
    x_px: float | None    # pixel x on the padded square frame (None if invalid)
    y_px: float | None
    valid: bool           # real, in-track gaze sample (vs pad / dropout)
    in_frame: bool        # gaze fell inside the source frame FOV
    x_norm: float | None  # provenance: normalized [0,1]
    y_norm: float | None
    x_1000: int | None    # provenance: 0-1000 bin
    y_1000: int | None


def px_to_padded_px(pad: PadTransform, x_src: float, y_src: float) -> tuple[float, float]:
    """Source-frame pixel -> pixel on the padded ``side``x``side`` frame.

    Reuses PadTransform's scale/offset (the exact ffmpeg scale-to-fit + centre-pad),
    returning pixel coords (norm * side) rather than normalized.
    """
    x_norm, y_norm = pad.px_to_norm(x_src, y_src)
    return clamp01(x_norm) * pad.side, clamp01(y_norm) * pad.side


def sample_segment_frames(
    grid_times: list[float],
    grid_px: list[float | None],
    grid_py: list[float | None],
    grid_in: list[bool],
    grid_valid: list[bool],
    *,
    seg_start_s: float,
    seg_end_s: float,
    fps: float,
    n_frames: int,
    pad: PadTransform,
) -> tuple[list[FrameGaze], int]:
    """Sample one gaze point per REAL video frame (gaze_hz == video_fps, 1:1).

    The on-disk clip is encoded at the canonical ``fps`` and has exactly ``n_frames``
    frames (probed from the mp4 by the caller, so the alignment is exact and never
    off-by-one). Video frame ``j`` is at clip-relative time ``j/fps`` == video-clock
    time ``seg_start_s + j/fps``; we read the gaze from the episode's canonical-fps
    grid (``grid_*``, video clock) at that frame's time and convert source px ->
    padded-frame px.

    There is NO cap and NO padding: the returned ``frames`` list has exactly
    ``n_frames`` entries, one per video frame. Frames whose gaze is invalid/out-of-FOV
    keep their slot (so frame<->point indices stay aligned) but carry no point. The
    second tuple element is the count of frames that carry a real (valid) gaze point.
    """
    if n_frames <= 0 or not grid_times:
        return [], 0
    n_grid = len(grid_times)
    dt = 1.0 / fps
    # grid index of video-clock time seg_start (the grid is uniform at i*dt from 0).
    base = int(round(seg_start_s * fps))

    frames: list[FrameGaze] = []
    num_real = 0
    for j in range(n_frames):
        gi = base + j
        t_rel = round(j * dt, 3)
        if 0 <= gi < n_grid:
            xs, ys = grid_px[gi], grid_py[gi]
            valid = bool(grid_valid[gi] and xs is not None and ys is not None)
        else:
            xs = ys = None
            valid = False
        if valid:
            x_pad, y_pad = px_to_padded_px(pad, xs, ys)
            xn, yn = pad.px_to_norm(xs, ys)
            xn, yn = clamp01(xn), clamp01(yn)
            frames.append(FrameGaze(
                t_s=t_rel, x_px=round(x_pad, 1), y_px=round(y_pad, 1),
                valid=True, in_frame=bool(grid_in[gi]),
                x_norm=round(xn, 6), y_norm=round(yn, 6),
                x_1000=to_bins(xn), y_1000=to_bins(yn),
            ))
            num_real += 1
        else:
            frames.append(FrameGaze(
                t_s=t_rel, x_px=None, y_px=None,
                valid=False, in_frame=False,
                x_norm=None, y_norm=None, x_1000=None, y_1000=None,
            ))
    return frames, num_real


def build_molmo2_row(
    *,
    dataset: str,
    episode_id: str,
    seg_index: int,
    video_rel: str,
    seg_start_s: float,
    seg_end_s: float,
    frames: list[FrameGaze],
    num_real: int,
    fps: float,
    side: int,
    annotation_text: str | None,
    prompt: str,
    annotation_channel: str | None = None,
    auxiliary_annotations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Assemble one Molmo2VideoPoint-style manifest row for a clip segment.

    gaze_hz == video_fps: ``points`` has exactly one entry per VIDEO FRAME (1:1, no
    cap, no padding); ``points[j]`` is the gaze at frame ``j``, as a raw PIXEL
    ``{"x","y"}`` on the ``side`` frame (the form ``Molmo2VideoPoint`` reads), or an
    empty list when that frame's gaze is invalid/out-of-FOV (masked from loss). The
    training label is ``points[0]`` (t0 gaze) or the first chunk ``points[:k]``.

    ``num_frames`` == ``len(points)`` (variable per clip); ``frame_mask[j]`` is 1 iff
    frame ``j`` carries a real gaze point, 0 otherwise; ``num_frames_real`` counts the
    1s. Normalized + 0-1000 forms are kept under ``provenance`` for validation.
    """
    points: list[list[dict[str, float]]] = []   # per-frame list of {x,y} (pixel)
    timestamps: list[float] = []
    frame_mask: list[int] = []
    prov_norm: list[list[float] | None] = []
    prov_1000: list[list[int] | None] = []
    for fg in frames:
        timestamps.append(fg.t_s)
        if fg.valid and fg.x_px is not None and fg.y_px is not None:
            points.append([{"x": fg.x_px, "y": fg.y_px}])
            prov_norm.append([fg.x_norm, fg.y_norm])
            prov_1000.append([fg.x_1000, fg.y_1000])
            frame_mask.append(1)
        else:
            points.append([])              # no gaze label for this frame
            prov_norm.append(None)
            prov_1000.append(None)
            frame_mask.append(0)

    answer = _format_answer(points, timestamps)
    user_text = prompt if not annotation_text else f"{prompt} (context: {annotation_text})"

    return {
        "id": f"{dataset}:{episode_id}#seg{seg_index}",
        "dataset": dataset,
        "episode_id": episode_id,
        "seg_index": seg_index,
        "style": "video_point",
        "video": video_rel,
        "message_list": [
            {"role": "user", "content": [
                {"type": "text", "text": user_text},
                {"type": "video", "video": video_rel},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": answer},
            ]},
        ],
        "points": points,                 # list-per-frame of {x,y} pixel on `side` frame (1:1 w/ video frames)
        "timestamps": timestamps,         # clip-relative, 1/fps grid (== frame j at j/fps)
        "frame_mask": frame_mask,         # 1=frame carries a real gaze point, 0=masked (no/invalid gaze)
        "num_frames_real": num_real,      # count of frames with a real gaze point
        "num_frames": len(points),        # total frames == total points (gaze_hz == video_fps)
        "fps": fps,
        "resolution": side,
        "metadata": {
            "clip_start_time": round(seg_start_s, 3),
            "clip_end_time": round(seg_end_s, 3),
            "annotation_text": annotation_text,         # DEFAULT annotation (drove this clip)
            "annotation_channel": annotation_channel,
            # AUXILIARY annotations: every other channel temporally covering this clip
            # (item 4). Each: {channel, text, start_s, end_s (clip-relative), overlap_s}.
            "auxiliary_annotations": auxiliary_annotations or [],
        },
        "provenance": {
            "points_norm": prov_norm,     # per-frame [x_norm,y_norm] in [0,1] or null
            "points_1000": prov_1000,     # per-frame [x_1000,y_1000] or null
            "label": "gaze",
        },
    }


def _format_answer(points: list[list[dict[str, float]]], timestamps: list[float]) -> str:
    """Human-readable assistant answer enumerating per-frame gaze points.

    For pure ``video_point`` style Molmo2 templates the exact wording downstream
    in its ``data_formatter.py``; we still emit a faithful, machine-parseable answer
    so the manifest is self-contained and inspectable.
    """
    parts = []
    for t, pts in zip(timestamps, points):
        if pts:
            p = pts[0]
            parts.append(f"t={t:.1f}s: ({p['x']:.0f}, {p['y']:.0f})")
        else:
            parts.append(f"t={t:.1f}s: (no gaze)")
    return "Gaze per frame -> " + "; ".join(parts)


MOLMO2_SCHEMA: dict[str, Any] = {
    "$comment": "Molmo2VideoPoint-style gaze finetuning rows (allenai/molmo2).",
    "fields": {
        "id": "<dataset>:<episode_id>#seg<k>",
        "style": "always 'video_point'",
        "video": "relative path to the side x side @ fps mp4 segment clip",
        "message_list": "chat: user{text, {type:video}} / assistant{text answer}",
        "points": "list-per-frame of [{x,y}] RAW PIXEL on the padded `resolution` frame; [] = no/invalid gaze that frame. EXACTLY ONE entry per video frame (1:1).",
        "timestamps": "list[float] clip-relative seconds; frame j is at j/fps",
        "frame_mask": "list[int] 1=frame carries a real gaze point, 0=masked (no/invalid gaze)",
        "num_frames_real": "count of frames with a real gaze point (sum of frame_mask)",
        "num_frames": "total frames == total points (variable per clip)",
        "fps": "canonical sampling fps == gaze hz (all datasets resampled down to this)",
        "resolution": "square side (378 for Molmo2)",
        "metadata": "{clip_start_time, clip_end_time, annotation_text (default), annotation_channel, auxiliary_annotations:[{channel,text,start_s,end_s,overlap_s}]}",
        "provenance": "{points_norm [0,1], points_1000 0-1000, label} per frame",
    },
    "notes": [
        "gaze_hz == video_fps: one gaze point per real video frame (1:1), variable length, NO cap and NO padding.",
        "Points are PIXEL on the padded resolution frame (Molmo2VideoPoint reads raw x,y; model tokenizes internally).",
        "Training label = points[0] (t0 gaze) or first chunk points[:k]; predict t0 given future frames + prompt.",
        "Frames with invalid/out-of-frame gaze have points=[] and frame_mask=0 (masked from loss) but keep their slot so frame<->point indices stay aligned.",
        "Molmo's fixed max_frames / padding-to-max conventions are punted (see --max-frames optional cap).",
    ],
}
