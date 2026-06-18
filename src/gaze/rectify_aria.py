"""Aria FISHEYE624 -> linear/pinhole rectification for clip extraction.

Egocentric Aria preview mp4s are fisheye-distorted (FISHEYE624, native 2880x2880,
``make_upright``-corrected). The base pipeline keeps video and gaze mutually
consistent *in that distorted space*: gaze is projected with
``get_gaze_vector_reprojection(..., make_upright=True)`` and the clip is a straight
ffmpeg pass over the distorted frames. This module adds an OPTIONAL step that warps
each clip frame to a linear (pinhole) camera and projects gaze through the SAME
linear calibration, so the per-frame gaze point still lands on the right pixel.

Design (see plan):
  * One representative online-calibration line per EPISODE drives both the image
    remap and the gaze projection -> any intra-episode calib drift is common-mode
    (it shifts the rendered pixel and the dot together), so alignment stays exact.
  * The 90 deg "upright" rotation that ``make_upright=True`` does internally is
    applied ONCE here via ``rotate_camera_calib_cw90deg`` to the source fisheye
    calib; the linear target is built from that rotated calib so the rectified
    frame and the linearly-projected gaze share one orientation. We then project
    gaze with ``make_upright=False`` (rotation already baked into the calib).
  * Remap maps are built once per episode at the preview-mp4 resolution and reused
    for every clip (cv2.remap). The output linear image keeps the source preview
    dims, so the downstream ``PadTransform`` geometry is unchanged.

``projectaria_tools`` and ``cv2`` are imported lazily so the base environment (which
lacks projectaria) never imports them unless ``--rectify`` runs on an Aria recipe.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

NATIVE = 2880  # Aria camera-rgb native square resolution (see curate_readers._NATIVE)


# --------------------------------------------------------------------------- #
# Orientation. Aria's camera-rgb is NATIVE 2880x2880; the preview mp4 and the
# validated gaze projection are both in the make_upright frame. Empirically (probed
# against get_gaze_vector_reprojection make_upright=True vs False) the upright<->native
# pixel map is a cw90 relabel:
#     upright = (NATIVE-1 - native_y, native_x)
#     native  = (upright_y, NATIVE-1 - upright_x)
# So the linear target calib is built UNROTATED (native), gaze is projected with
# make_upright=True (matches GT to ~8 px), and the remap composes the rotations so the
# OUTPUT frame is upright-linear, sampling the upright-fisheye preview mp4.
# --------------------------------------------------------------------------- #
def _upright_to_native(ux: float, uy: float) -> tuple[float, float]:
    return (uy, (NATIVE - 1) - ux)


def _native_to_upright(nx: float, ny: float) -> tuple[float, float]:
    return ((NATIVE - 1) - ny, nx)


@dataclass
class AriaRectifier:
    """Per-episode linear-rectification state (calibs + cv2 remap maps).

    Built once from a representative ``online_calibration.jsonl`` line via
    :func:`build_linear_rectifier`. ``dst_lin`` is the NATIVE pinhole target used for
    BOTH the image remap and gaze projection; orientation handling (make_upright) is
    composed into the remap and the ``make_upright=True`` gaze projection.
    """

    device_calib: Any            # projectaria DeviceCalibration (CPF transform source)
    dst_lin: Any                 # linear (pinhole) CameraCalibration, NATIVE orientation/res
    prev_scale: float            # preview_mp4_w / NATIVE
    out_w: int                   # rectified output width  (== source preview width)
    out_h: int                   # rectified output height (== source preview height)
    _maps: Any = field(default=None, repr=False)  # (map_x, map_y) float32, lazily built


def build_linear_rectifier(
    calib_line: dict[str, Any], *, preview_w: int, preview_h: int, focal_scale: float = 1.0,
) -> AriaRectifier:
    """Build the per-episode rectifier from one online-calibration line.

    ``preview_w/h`` are the source preview mp4 dims. ``focal_scale`` scales the target
    linear focal length: ``<1`` widens the FOV (less periphery cropped, more stretch),
    ``>1`` zooms in. ``1.0`` keeps the source focal.
    """
    from projectaria_tools.core.calibration import get_linear_camera_calibration
    from . import curate_readers as cr

    dc, rgb_cam = cr._build_aria_device(calib_line)
    focal = float(rgb_cam.get_focal_lengths()[0]) * float(focal_scale)
    lin = get_linear_camera_calibration(
        NATIVE, NATIVE, focal, "camera-rgb", rgb_cam.get_transform_device_camera(),
    )
    prev_scale = (preview_w / float(NATIVE)) if preview_w else 1.0
    rect = AriaRectifier(
        device_calib=dc, dst_lin=lin, prev_scale=prev_scale,
        out_w=int(preview_w or NATIVE), out_h=int(preview_h or NATIVE),
    )
    rect._maps = _build_remap_maps(rgb_cam, lin, rect)
    return rect


def _build_remap_maps(src_fisheye, dst_lin, rect: AriaRectifier):
    """cv2 remap maps: for each UPRIGHT-LINEAR output pixel, the UPRIGHT-fisheye preview
    pixel to sample. Composes the make_upright cw90 relabel on both ends so the output
    frame matches the preview mp4 orientation and the make_upright=True gaze projection.

    Built at preview resolution (output grid == source preview dims).
    """
    import numpy as np

    s = rect.prev_scale
    ow, oh = rect.out_w, rect.out_h
    inv_s = (1.0 / s) if s else 1.0
    map_x = np.full((oh, ow), -1.0, dtype=np.float32)
    map_y = np.full((oh, ow), -1.0, dtype=np.float32)
    for vy in range(oh):
        for ux in range(ow):
            # output upright-linear preview px -> native upright-linear px (2880 grid)
            uX, uY = ux * inv_s, vy * inv_s
            # upright-linear -> native-linear, unproject -> ray, project -> native fisheye
            nlx, nly = _upright_to_native(uX, uY)
            ray = dst_lin.unproject(np.array([nlx, nly], dtype=float))
            if ray is None:
                continue
            fish = src_fisheye.project(ray)
            if fish is None:
                continue
            # native fisheye -> upright fisheye -> preview px
            ufx, ufy = _native_to_upright(float(fish[0]), float(fish[1]))
            map_x[vy, ux] = ufx * s
            map_y[vy, ux] = ufy * s
    return map_x, map_y


def project_gaze_linear(row: dict[str, Any], rect: AriaRectifier) -> tuple[float, float] | None:
    """Project one CPF gaze row through the rectifier's linear calib -> upright preview px.

    Mirrors ``overlay._project_aria_one`` but targets the per-episode linear calib with
    ``make_upright=True`` so the point lands in the same upright-linear space as the
    rectified clip frame. Returns ``None`` for missing/out-of-FOV gaze.
    """
    import numpy as np
    from projectaria_tools.core.mps import EyeGaze
    from projectaria_tools.core.mps.utils import get_gaze_vector_reprojection

    ly, ry, pitch = row.get("left_yaw"), row.get("right_yaw"), row.get("pitch")
    if ly is None or ry is None or pitch is None:
        return None
    depth = row.get("depth") or 1.0
    eg = EyeGaze()
    eg.yaw = 0.5 * (ly + ry)
    eg.pitch = pitch
    eg.depth = depth
    px = get_gaze_vector_reprojection(
        eg, "camera-rgb", rect.device_calib, rect.dst_lin, depth_m=depth, make_upright=True,
    )
    if px is None:
        return None
    px = np.asarray(px) * rect.prev_scale
    return (float(px[0]), float(px[1]))


def _probe_dims(path: Path) -> tuple[int, int]:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", str(path)],
        capture_output=True, text=True,
    )
    try:
        w, h = (r.stdout or "").strip().split("x")[:2]
        return int(w), int(h)
    except (ValueError, IndexError):
        return 0, 0


def rectify_clip_video(
    full_src: Path,
    out_path: Path,
    rect: AriaRectifier,
    *,
    start_s: float,
    seg_len: float,
    fps: float,
    side: int,
    reuse_existing: bool = False,
) -> dict[str, Any]:
    """Trim+rectify+resample one clip: ffmpeg trim/fps -> cv2.remap -> ffmpeg scale/pad.

    Same vmeta contract as ``training.resample_segment_from_local`` (incl. probed
    ``nb_frames`` so the 1:1 frame<->point map holds). The middle stage warps each
    decoded frame through the precomputed linear-rectification maps.
    """
    import numpy as np
    import cv2

    from . import overlay as ov
    from .training import _probe_nframes

    if reuse_existing and out_path.exists():
        meta = ov._probe(out_path)
        nf = _probe_nframes(out_path)
        if nf > 0 and int(meta.get("width") or 0) == side and int(meta.get("height") or 0) == side:
            return {"path": str(out_path), "width": meta["width"], "height": meta["height"],
                    "fps": fps, "duration_s": meta.get("duration"), "start_s": start_s,
                    "nb_frames": nf, "reused": True, "rectified": True}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sw, sh = _probe_dims(full_src)
    if sw <= 0 or sh <= 0:
        raise RuntimeError(f"rectify: could not probe source dims for {full_src}")
    map_x, map_y = rect._maps

    # Stage 1: decode the [start_s, start_s+seg_len] window at the canonical fps as
    # raw bgr frames (dims unchanged by the fps filter).
    dec = subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{start_s:.3f}", "-i", str(full_src), "-t", f"{seg_len:.3f}",
         "-vf", f"fps={fps}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        capture_output=True,
    )
    if dec.returncode != 0:
        raise RuntimeError(f"rectify decode failed: {dec.stderr[-600:].decode('utf-8', 'replace')}")
    frame_bytes = sw * sh * 3
    raw = dec.stdout
    n = len(raw) // frame_bytes if frame_bytes else 0
    if n <= 0:
        raise RuntimeError("rectify: decoded 0 frames")

    # Stage 2: remap each frame to the linear camera, then pipe to the encoder with the
    # SAME scale+pad vf as the distorted path so the pad geometry is byte-identical.
    vf = (
        f"scale={side}:{side}:force_original_aspect_ratio=decrease,"
        f"pad={side}:{side}:(ow-iw)/2:(oh-ih)/2"
    )
    enc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-nostats",
         "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{rect.out_w}x{rect.out_h}", "-r", f"{fps}", "-i", "-",
         "-vf", vf, "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    assert enc.stdin is not None
    arr = np.frombuffer(raw, dtype=np.uint8, count=n * frame_bytes).reshape(n, sh, sw, 3)
    err = b""
    try:
        for i in range(n):
            rectified = cv2.remap(arr[i], map_x, map_y, interpolation=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            enc.stdin.write(rectified.tobytes())
        enc.stdin.close()  # signal EOF; encoder flushes and exits
        err = enc.stderr.read() if enc.stderr else b""
    finally:
        enc.wait()
    if enc.returncode != 0 or not out_path.exists():
        raise RuntimeError(f"rectify encode failed: {err[-600:].decode('utf-8', 'replace')}")

    from . import overlay as ov2
    meta = ov2._probe(out_path)
    return {"path": str(out_path), "width": meta["width"], "height": meta["height"],
            "fps": fps, "duration_s": meta.get("duration"), "start_s": start_s,
            "nb_frames": _probe_nframes(out_path), "rectified": True}


def pick_representative_calib(calib_lines: list[dict[str, Any]]) -> dict[str, Any]:
    """Median-timestamp online-calibration line (drives the whole-episode remap)."""
    if not calib_lines:
        raise ValueError("no calibration lines")
    ordered = sorted(
        calib_lines, key=lambda c: c.get("tracking_timestamp_us") or 0.0,
    )
    return ordered[len(ordered) // 2]
