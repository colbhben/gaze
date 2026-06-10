"""MolmoAct2 / Molmo2 video-point finetuning manifest emitter.

Builds ``Molmo2VideoPoint``-style training rows (confirmed against
``allenai/molmo2`` ``olmo/data/molmo2_datasets.py``) for the gaze task:
predict where the camera wearer is looking, per frame, over a short video clip.

Confirmed MolmoAct2 / Molmo2 video-point format (from source):
  * multi-frame video; video pointing standardizes on 2 fps (0.5 s grid);
  * per-frame frames preprocessed to 378x378 aspect-preserving (resize+pad);
  * points stored as RAW PIXEL ``{"x": float, "y": float}`` (one list per frame),
    NOT normalized and NOT <point> tokens (the model tokenizes internally);
  * ``timestamps`` clip-relative on the ``1/fps`` grid;
  * example = ``message_list`` chat (user: text + {"type":"video"}; assistant: answer)
    with ``metadata{clip_start_time, clip_end_time}`` and ``style:"video_point"``.

Design decisions (user-confirmed):
  * per-frame gaze targets (gaze is dynamic);
  * canonical fps is user-defined; ALL datasets resample DOWN to it (sources are
    24-30 fps, always >= canonical);
  * variable-length clips PADDED to ``max_frames`` for a consistent batch tensor
    shape, with a per-frame ``frame_mask`` (1=real, 0=pad) and ``num_frames_real``;
    padded frames repeat the last real frame and carry ``valid:false`` points so
    loss/attention ignore them.

We keep the normalized ``[0,1]`` and integer ``0-1000`` forms per frame in a
``provenance`` sidecar (user: "match MolmoAct2 + keep normalized").
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Reuse the validated pad geometry from the contract base (no circular import).
from .curate import PadTransform, clamp01, to_bins


DEFAULT_PROMPT = "Point to where the camera wearer is looking."
DEFAULT_FPS = 2.0
DEFAULT_MAX_FRAMES = 8
DEFAULT_RESOLUTION = 378
# MolmoAct2 video pointing asserts timestamps on a 0.5 s grid; 1/fps must divide it.
TIMESTAMP_GRID_S = 0.5


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
    max_frames: int,
    pad: PadTransform,
) -> tuple[list[FrameGaze], int]:
    """Sample one clip segment's per-frame gaze, capped + padded to ``max_frames``.

    ``grid_*`` is the episode's gaze resampled onto the canonical fps grid (video
    clock). We take grid points in ``[seg_start_s, seg_end_s)``, cap to ``max_frames``
    (uniformly subsampling if longer), convert source px -> padded-frame px, then PAD
    up to ``max_frames`` by repeating the last real frame with ``valid=False``.

    Returns ``(frames, num_real)`` where ``frames`` has exactly ``max_frames`` entries
    and the first ``num_real`` are real (mask=1), the rest padding (mask=0).
    """
    # indices of grid points inside the segment
    idxs = [i for i, t in enumerate(grid_times) if seg_start_s <= t < seg_end_s]
    if not idxs:
        return [], 0
    # cap: uniformly subsample to max_frames if the segment yields more
    if len(idxs) > max_frames:
        step = len(idxs) / max_frames
        idxs = [idxs[int(k * step)] for k in range(max_frames)]

    frames: list[FrameGaze] = []
    for i in idxs:
        # clip-relative time snapped to the 1/fps grid
        t_rel = round((grid_times[i] - seg_start_s) / (1.0 / fps)) * (1.0 / fps)
        xs, ys = grid_px[i], grid_py[i]
        valid = bool(grid_valid[i] and xs is not None and ys is not None)
        if valid:
            x_pad, y_pad = px_to_padded_px(pad, xs, ys)
            xn, yn = pad.px_to_norm(xs, ys)
            xn, yn = clamp01(xn), clamp01(yn)
            frames.append(FrameGaze(
                t_s=round(t_rel, 3), x_px=round(x_pad, 1), y_px=round(y_pad, 1),
                valid=True, in_frame=bool(grid_in[i]),
                x_norm=round(xn, 6), y_norm=round(yn, 6),
                x_1000=to_bins(xn), y_1000=to_bins(yn),
            ))
        else:
            frames.append(FrameGaze(
                t_s=round(t_rel, 3), x_px=None, y_px=None,
                valid=False, in_frame=False,
                x_norm=None, y_norm=None, x_1000=None, y_1000=None,
            ))
    num_real = len(frames)

    # pad up to max_frames by repeating the last real frame (mask handled by caller)
    if num_real and num_real < max_frames:
        last = frames[-1]
        dt = 1.0 / fps
        for k in range(num_real, max_frames):
            frames.append(FrameGaze(
                t_s=round(last.t_s + (k - num_real + 1) * dt, 3),
                x_px=last.x_px, y_px=last.y_px,
                valid=False, in_frame=False,
                x_norm=last.x_norm, y_norm=last.y_norm,
                x_1000=last.x_1000, y_1000=last.y_1000,
            ))
    return frames, num_real


def build_molmoact_row(
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
    max_frames: int,
    side: int,
    annotation_text: str | None,
    prompt: str,
    annotation_channel: str | None = None,
) -> dict[str, Any]:
    """Assemble one Molmo2VideoPoint-style manifest row for a clip segment.

    The on-disk gaze target is per-frame raw PIXEL ``{"x","y"}`` on the ``side`` frame
    (the form ``Molmo2VideoPoint`` reads). ``frame_mask`` + ``num_frames_real`` carry
    the variable-length-padded-to-``max_frames`` structure for batching. Normalized +
    0-1000 forms are kept under ``provenance`` for portability/validation.
    """
    points: list[list[dict[str, float]]] = []   # per-frame list of {x,y} (pixel)
    timestamps: list[float] = []
    frame_mask: list[int] = []
    prov_norm: list[list[float] | None] = []
    prov_1000: list[list[int] | None] = []
    for k, fg in enumerate(frames):
        timestamps.append(fg.t_s)
        is_real = k < num_real and fg.valid
        frame_mask.append(1 if k < num_real else 0)
        if fg.x_px is not None and fg.y_px is not None and is_real:
            points.append([{"x": fg.x_px, "y": fg.y_px}])
            prov_norm.append([fg.x_norm, fg.y_norm])
            prov_1000.append([fg.x_1000, fg.y_1000])
        else:
            points.append([])              # no gaze label for this frame
            prov_norm.append(None)
            prov_1000.append(None)

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
        "points": points,                 # list-per-frame of {x,y} pixel on `side` frame
        "timestamps": timestamps,         # clip-relative, 1/fps grid
        "frame_mask": frame_mask,         # 1=real frame, 0=pad
        "num_frames_real": num_real,
        "num_frames": max_frames,
        "fps": fps,
        "resolution": side,
        "metadata": {
            "clip_start_time": round(seg_start_s, 3),
            "clip_end_time": round(seg_end_s, 3),
            "annotation_text": annotation_text,
            "annotation_channel": annotation_channel,
        },
        "provenance": {
            "points_norm": prov_norm,     # per-frame [x_norm,y_norm] in [0,1] or null
            "points_1000": prov_1000,     # per-frame [x_1000,y_1000] or null
            "label": "gaze",
        },
    }


def _format_answer(points: list[list[dict[str, float]]], timestamps: list[float]) -> str:
    """Human-readable assistant answer enumerating per-frame gaze points.

    For pure ``video_point`` style MolmoAct2 templates the exact wording downstream
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


MOLMOACT_SCHEMA: dict[str, Any] = {
    "$comment": "Molmo2VideoPoint-style gaze finetuning rows (allenai/molmo2).",
    "fields": {
        "id": "<dataset>:<episode_id>#seg<k>",
        "style": "always 'video_point'",
        "video": "relative path to the side x side @ fps mp4 segment clip",
        "message_list": "chat: user{text, {type:video}} / assistant{text answer}",
        "points": "list-per-frame of [{x,y}] RAW PIXEL on the padded `resolution` frame; [] = no gaze that frame",
        "timestamps": "list[float] clip-relative seconds on the 1/fps grid",
        "frame_mask": "list[int] 1=real frame, 0=pad-to-max_frames",
        "num_frames_real": "count of real (non-pad) frames",
        "num_frames": "max_frames (padded tensor length)",
        "fps": "canonical sampling fps (all datasets resampled down to this)",
        "resolution": "square side (378 for MolmoAct2)",
        "metadata": "{clip_start_time, clip_end_time, annotation_text}",
        "provenance": "{points_norm [0,1], points_1000 0-1000, label} per frame",
    },
    "notes": [
        "Points are PIXEL on the padded resolution frame (Molmo2VideoPoint reads raw x,y; model tokenizes internally).",
        "Variable-length clips padded to num_frames; use frame_mask + num_frames_real to ignore padding in loss.",
        "Per-frame targets; frames with invalid/out-of-frame gaze have points=[] (masked from loss).",
    ],
}
