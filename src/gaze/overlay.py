"""Gaze + annotation OVERLAY renderer for the curation pipeline.

Builds on the extraction harness (``curate.py`` / ``curate_readers.py``):

  * Pulls one episode's egocentric mp4 (trims to a short clip first, so the
    huge nymeria/hd-epic mp4s never sit on local disk in full longer than the
    one trim step).
  * For each frame: maps frame index -> seconds, reconciles onto the common
    video-zero clock per the recipe ``epoch_sync`` block, picks the nearest
    gaze sample, PROJECTS it to mp4 pixels (reusing the validated projection
    math from ``curate_readers``), and draws a gaze dot/crosshair.
  * For each frame: finds the annotation segment(s) active at ``t`` per channel
    and draws a channel-coloured, word-wrapped caption band.
  * Encodes the drawn PNG frames to a small (<=720p, <=max_seconds) mp4 with
    ffmpeg, optionally hstacked with a ground-truth reference video.

The per-frame projection is factored as ``ProjectionContext`` (loads
calibration/poses ONCE) + ``project_one`` so we never reload calib per frame.

Pure-stdlib epoch/lookup helpers (``frame_time``, ``reconcile_to_video_clock``,
``nearest_gaze_index``, ``active_segments``) are import-light and unit-tested
in ``tests/test_overlay.py`` with synthetic data (no remote / video needed).
"""
from __future__ import annotations

import bisect
import json
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from . import curate_readers as cr
from .curate import (
    EpisodeBundle,
    GazeTable,
    Puller,
    VideoMeta,
    load_recipe,
)

# --------------------------------------------------------------------------- #
# Channel colours (RGB) for caption bands + gaze dot.
# --------------------------------------------------------------------------- #
GAZE_COLOR = (255, 60, 60)            # red crosshair
CHANNEL_COLORS = [
    (90, 200, 250),                   # cyan
    (255, 210, 80),                   # amber
    (130, 230, 130),                  # green
    (220, 150, 250),                  # violet
    (250, 160, 110),                  # orange
]


# =========================================================================== #
# 1. EPOCH RECONCILIATION + LOOKUP (pure stdlib, unit-tested)
# =========================================================================== #
def frame_time(frame_index: int, fps: float) -> float:
    """Seconds of mp4 frame ``frame_index`` on the *video* clock (frame 0 == 0s)."""
    if not fps:
        raise ValueError("fps required")
    return frame_index / fps


def reconcile_to_video_clock(
    raw_t: float | None,
    transform: str,
    *,
    gaze_t0: float | None = None,
    clip_start_s: float | None = None,
) -> float | None:
    """Map a gaze/annotation timestamp onto the common video-zero clock.

    The recipe ``epoch_sync`` per-channel ``transform`` decides how:
      * ``as_is``               -> value is already video-zero seconds.
      * ``subtract_first_gaze`` -> value is on the device clock that starts at the
                                   first gaze sample; subtract ``gaze_t0``.
      * ``subtract_clip_start`` -> value is session-absolute seconds; subtract the
                                   clip's start time (``clip_start_s``).
    """
    if raw_t is None:
        return None
    if transform == "as_is":
        return raw_t
    if transform == "subtract_first_gaze":
        if gaze_t0 is None:
            raise ValueError("subtract_first_gaze needs gaze_t0")
        return raw_t - gaze_t0
    if transform == "subtract_clip_start":
        if clip_start_s is None:
            raise ValueError("subtract_clip_start needs clip_start_s")
        return raw_t - clip_start_s
    raise ValueError(f"unknown epoch transform: {transform!r}")


def nearest_gaze_index(
    gaze_times_video: Sequence[float | None],
    t: float,
    *,
    max_dt: float | None = None,
    valid_mask: Sequence[bool] | None = None,
) -> int | None:
    """Index of the gaze sample whose video-clock time is nearest ``t``.

    ``gaze_times_video`` is gaze sample times ALREADY reconciled to the video
    clock (ascending, ``None`` entries skipped). Returns ``None`` when the table
    is empty or the nearest sample is farther than ``max_dt`` seconds away
    (so a frame with no nearby gaze draws no dot rather than a stale one).

    When ``valid_mask`` is given, samples whose mask entry is falsey are
    excluded entirely: invalid/dropout rows (e.g. egoexolearn's (0,0)/(0.5,0.5)
    placeholders, or a Tobii tracking dropout) are never selected, and a frame
    whose only nearby samples are invalid draws no dot rather than a stale or
    center-snapped one. Combined with ``max_dt``, this means an invalid run
    longer than ``max_dt`` correctly yields a blank gaze for those frames.
    """
    # Build a parallel (time, index) list excluding Nones (and invalids), ascending.
    cleaned = [
        (tv, i)
        for i, tv in enumerate(gaze_times_video)
        if tv is not None and (valid_mask is None or (i < len(valid_mask) and valid_mask[i]))
    ]
    if not cleaned:
        return None
    times = [tv for tv, _ in cleaned]
    pos = bisect.bisect_left(times, t)
    best_i, best_d = None, None
    for cand in (pos - 1, pos, pos + 1):
        if 0 <= cand < len(cleaned):
            d = abs(times[cand] - t)
            if best_d is None or d < best_d:
                best_d, best_i = d, cleaned[cand][1]
    if best_i is None:
        return None
    if max_dt is not None and best_d is not None and best_d > max_dt:
        return None
    return best_i


def active_segments(
    segments: Sequence[dict[str, Any]],
    t: float,
    *,
    kind: str,
    point_tol: float = 0.5,
) -> list[dict[str, Any]]:
    """Segments active at video-clock time ``t`` for one channel.

    ``segments`` are expected ALREADY on the video clock (``start_s``/``end_s``
    /``point_s`` rebased). For ``interval`` channels, returns every segment whose
    ``[start_s, end_s]`` covers ``t``. For ``point`` channels, returns the single
    nearest point within ``point_tol`` seconds (so a momentary label briefly
    shows). Missing/None bounds are skipped (no crash on gaps).
    """
    out: list[dict[str, Any]] = []
    if kind == "point":
        best, best_d = None, None
        for s in segments:
            p = s.get("point_s")
            if p is None:
                continue
            d = abs(p - t)
            if (best_d is None or d < best_d) and d <= point_tol:
                best, best_d = s, d
        return [best] if best is not None else []
    for s in segments:
        start = s.get("start_s")
        end = s.get("end_s")
        if start is None:
            continue
        if end is None:
            # open interval / point-as-interval: show for point_tol seconds
            if start <= t <= start + point_tol:
                out.append(s)
            continue
        if start <= t <= end:
            out.append(s)
    return out


# =========================================================================== #
# 2. EPISODE BUNDLE LOADING (reuse pre-extracted /tmp bundles when present)
# =========================================================================== #
EXTRACT_DIR = Path("/tmp/gaze_extract")


def load_full_bundle(slug: str) -> dict[str, Any] | None:
    """Load a pre-extracted ``<slug>_full.json`` (gaze rows + anno segments)."""
    p = EXTRACT_DIR / f"{slug}_full.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


@dataclass
class EpisodeData:
    """The data the renderer needs, pulled from a full bundle or fresh extract."""

    slug: str
    episode_id: str
    video: dict[str, Any]
    gaze_space: str
    gaze_rows: list[dict[str, Any]]
    annotations: list[dict[str, Any]]      # [{name, kind, segments:[...]}, ...]
    epoch_sync: dict[str, Any]
    projection_method: str
    projection: dict[str, Any]             # gaze_format.projection (with _frame_dims)
    recipe: dict[str, Any]
    tok: dict[str, str]


def build_episode_data(
    slug: str,
    episode_id: str,
    puller: Puller,
    *,
    sample_extra: dict[str, Any] | None = None,
    reuse: bool = True,
) -> EpisodeData:
    """Assemble :class:`EpisodeData`, preferring the cached full bundle."""
    recipe = load_recipe(slug)
    extra = dict(sample_extra or {})
    tok = cr._episode_tokens(slug, episode_id, extra)
    gf = recipe["gaze"]["gaze_format"]
    proj = dict(gf.get("projection") or {})
    if gf.get("frame_dims"):
        proj["_frame_dims"] = gf["frame_dims"]

    full = load_full_bundle(slug) if reuse else None
    if full is not None and full.get("episode_id") == episode_id and full.get("gaze"):
        return EpisodeData(
            slug=slug,
            episode_id=episode_id,
            video=full["video"],
            gaze_space=full["gaze"]["coordinate_space"],
            gaze_rows=full["gaze"]["rows"],
            annotations=[
                {"name": a["name"], "kind": a["kind"], "segments": a["segments"]}
                for a in full["annotations"]
            ],
            epoch_sync=recipe.get("epoch_sync", {}),
            projection_method=proj.get("method", "none"),
            projection=proj,
            recipe=recipe,
            tok=tok,
        )

    # Fallback: fresh extract (rare; the 7 samples are pre-extracted).
    bundle: EpisodeBundle = cr.extract_episode(slug, episode_id, puller, sample_extra=extra)
    if bundle.gaze is None or bundle.video is None:
        raise RuntimeError(f"{slug}:{episode_id} missing gaze/video ({bundle.emit_reason})")
    from dataclasses import asdict
    return EpisodeData(
        slug=slug,
        episode_id=episode_id,
        video=asdict(bundle.video),
        gaze_space=bundle.gaze.coordinate_space,
        gaze_rows=bundle.gaze.rows,
        annotations=[
            {"name": a.name, "kind": a.kind, "segments": a.segments}
            for a in bundle.annotations
        ],
        epoch_sync=recipe.get("epoch_sync", {}),
        projection_method=proj.get("method", "none"),
        projection=bundle.gaze.projection or proj,
        recipe=recipe,
        tok=tok,
    )


# --------------------------------------------------------------------------- #
# Reconcile gaze + annotations onto the video clock ONCE, up front.
# --------------------------------------------------------------------------- #
def gaze_t0_of(data: EpisodeData) -> float:
    rows = data.gaze_rows
    for r in rows:
        if r.get("t_s") is not None:
            return float(r["t_s"])
    return 0.0


def clip_start_of(slug: str, episode_id: str) -> float | None:
    """egtea: clip start seconds = filename start_ms / 1000."""
    if slug != "egtea":
        return None
    import re

    m = re.search(r"-(\d+)-(\d+)-F\d+-F\d+$", episode_id)
    if m:
        return int(m.group(1)) / 1000.0
    return None


def reconciled_gaze_times(data: EpisodeData) -> list[float | None]:
    """Gaze sample times on the video clock per epoch_sync.gaze.transform."""
    es = data.epoch_sync or {}
    transform = (es.get("gaze") or {}).get("transform", "as_is")
    g0 = gaze_t0_of(data)
    out: list[float | None] = []
    for r in data.gaze_rows:
        out.append(
            reconcile_to_video_clock(r.get("t_s"), transform, gaze_t0=g0)
        )
    return out


def reconciled_annotations(data: EpisodeData) -> list[dict[str, Any]]:
    """Annotation channels with segment times rebased onto the video clock."""
    es = data.epoch_sync or {}
    transform = (es.get("annotations") or {}).get("transform", "as_is")
    g0 = gaze_t0_of(data)
    clip_start = clip_start_of(data.slug, data.episode_id)
    out = []
    for ch in data.annotations:
        rebased = []
        for s in ch["segments"]:
            ns = dict(s)
            ns["start_s"] = reconcile_to_video_clock(
                s.get("start_s"), transform, gaze_t0=g0, clip_start_s=clip_start
            )
            ns["end_s"] = reconcile_to_video_clock(
                s.get("end_s"), transform, gaze_t0=g0, clip_start_s=clip_start
            )
            ns["point_s"] = reconcile_to_video_clock(
                s.get("point_s"), transform, gaze_t0=g0, clip_start_s=clip_start
            )
            rebased.append(ns)
        out.append({"name": ch["name"], "kind": ch["kind"], "segments": rebased})
    return out


# =========================================================================== #
# 3. PER-FRAME PROJECTION (ProjectionContext loads calib ONCE; project_one each frame)
# =========================================================================== #
@dataclass
class ProjectionContext:
    """Holds the (expensive) calibration/pose state for per-frame projection.

    Built once via :func:`build_projection_context`; ``project_one`` then maps a
    single gaze row to mp4 pixels reusing the validated math in curate_readers.
    """

    method: str
    width: int | None
    height: int | None
    # aria
    calib_lines: list[dict[str, Any]] | None = None
    aria_scale: float = 1.0
    # psi
    intr: dict[str, Any] | None = None
    poses: list[dict[str, Any]] | None = None
    psi_rescale: float = 1.0
    # normalize_by_dims
    frame_dims: tuple[float, float] | None = None
    coordinate_space: str | None = None
    notes: list[str] = field(default_factory=list)


def build_projection_context(data: EpisodeData, puller: Puller) -> ProjectionContext:
    method = data.projection_method
    w = data.video.get("width")
    h = data.video.get("height")
    notes: list[str] = []
    ctx = ProjectionContext(method=method, width=w, height=h,
                            coordinate_space=data.gaze_space, notes=notes)
    root = data.recipe["root"]
    tok = data.tok

    if method == "projectaria_cpf":
        proj = data.projection or {}
        calib_spec = proj.get("calibration") or {}
        calib_tmpl = calib_spec.get("file", "")
        local_tok = {**tok, "episode_id": tok.get("episode_id", data.episode_id)}
        if "{slam_index}" in calib_tmpl:
            idx = cr._deref_slam_index(puller, root, local_tok, notes)
            if idx is not None:
                local_tok["slam_index"] = idx
        calib_rel = f"{root}/{cr._fill(calib_tmpl, local_tok)}".replace("//", "/")
        calib_local = puller.pull(calib_rel)
        ctx.calib_lines = cr._load_calib_lines(calib_local)
        ctx.aria_scale = (w / 2880.0) if w else 1.0
        notes.append(f"aria scale = mp4_w/2880 = {ctx.aria_scale:.5f}; make_upright=True")

    elif method == "psi_pinhole_ray":
        proj = data.projection or {}
        calib_spec = proj.get("calibration") or {}
        extr_spec = proj.get("extrinsics") or {}
        intr_rel = f"{root}/{cr._fill(calib_spec.get('file',''), tok)}".replace("//", "/")
        pose_rel = f"{root}/{cr._fill(extr_spec.get('file',''), tok)}".replace("//", "/")
        ctx.intr = cr._parse_psi_intrinsics(puller.pull(intr_rel))
        ctx.poses = cr._parse_pose_sync(puller.pull(pose_rel))
        ctx.psi_rescale = (w / ctx.intr["w"]) if ctx.intr["w"] else 1.0
        # precompute pose times for fast nearest lookup
        ctx._pose_times = [p["t_s"] for p in ctx.poses]  # type: ignore[attr-defined]

    elif method in ("normalize_by_dims",):
        fd = (data.projection or {}).get("_frame_dims")
        if fd and len(fd) == 2:
            ctx.frame_dims = (float(fd[0]), float(fd[1]))
        else:
            ctx.frame_dims = (float(w or 1), float(h or 1))

    return ctx


def _nearest_pose_fast(ctx: ProjectionContext, t: float):
    poses = ctx.poses or []
    if not poses:
        return None
    times = getattr(ctx, "_pose_times", None)
    if times is None:
        return cr._nearest_pose(poses, t)
    pos = bisect.bisect_left(times, t)
    best, best_d = None, None
    for cand in (pos - 1, pos, pos + 1):
        if 0 <= cand < len(poses):
            d = abs(times[cand] - t)
            if best_d is None or d < best_d:
                best, best_d = poses[cand], d
    return best


def project_one(
    row: dict[str, Any], ctx: ProjectionContext
) -> tuple[float, float] | None:
    """Project a single gaze row to mp4 pixels. Returns (x_px, y_px) or None.

    Reuses the exact validated math from curate_readers per method; ``None`` for
    out-of-FOV / missing data / behind-camera so the caller draws no dot.
    """
    method = ctx.method
    w, h = ctx.width, ctx.height

    if method == "already_2d":
        x, y = row.get("x"), row.get("y")
        return (float(x), float(y)) if (x is not None and y is not None) else None

    if method == "normalize_by_dims":
        x, y = row.get("x"), row.get("y")
        if x is None or y is None:
            return None
        fdims = ctx.frame_dims or (float(w or 1), float(h or 1))
        if ctx.coordinate_space == "normalized_2d":
            return (x * (w or fdims[0]), y * (h or fdims[1]))
        sx = (w / fdims[0]) if (w and fdims[0]) else 1.0
        sy = (h / fdims[1]) if (h and fdims[1]) else 1.0
        return (x * sx, y * sy)

    if method == "projectaria_cpf":
        return _project_aria_one(row, ctx)

    if method == "psi_pinhole_ray":
        return _project_psi_one(row, ctx)

    return None


def _project_aria_one(row, ctx: ProjectionContext):
    import numpy as np
    from projectaria_tools.core.mps import EyeGaze
    from projectaria_tools.core.mps.utils import get_gaze_vector_reprojection

    if not ctx.calib_lines:
        return None
    ly, ry, pitch = row.get("left_yaw"), row.get("right_yaw"), row.get("pitch")
    if ly is None or ry is None or pitch is None:
        return None
    ts_us = row.get("_tracking_timestamp_us")
    cl = cr._nearest_calib(ctx.calib_lines, ts_us) if ts_us is not None else ctx.calib_lines[0]
    dc, rgb_cam = cr._build_aria_device(cl)
    depth = row.get("depth") or 1.0
    eg = EyeGaze()
    eg.yaw = 0.5 * (ly + ry)
    eg.pitch = pitch
    eg.depth = depth
    px = get_gaze_vector_reprojection(eg, "camera-rgb", dc, rgb_cam, depth_m=depth, make_upright=True)
    if px is None:
        return None
    px = np.asarray(px) * ctx.aria_scale
    return (float(px[0]), float(px[1]))


def _project_psi_one(row, ctx: ProjectionContext):
    import numpy as np

    intr = ctx.intr
    if intr is None:
        return None
    t = row.get("t_s")
    P = np.array([row.get("px"), row.get("py"), row.get("pz")], dtype=float)
    V = np.array([row.get("vx"), row.get("vy"), row.get("vz")], dtype=float)
    if t is None or np.any(np.isnan(P)) or np.any(np.isnan(V)):
        return None
    pose = _nearest_pose_fast(ctx, t)
    if pose is None:
        return None
    T = pose["T"]
    R = T[:3, :3]
    tr = T[:3, 3]
    pt_world = P + 1.0 * V
    pt_cam = R.T @ (pt_world - tr)
    X, Y, Z = pt_cam
    if X <= 0:
        return None
    u = (intr["ppx"] - intr["flx"] * (Y / X)) * ctx.psi_rescale
    v = (intr["ppy"] - intr["fly"] * (Z / X)) * ctx.psi_rescale
    return (float(u), float(v))


# =========================================================================== #
# 4. VIDEO PULL + TRIM (keep big mp4s off local disk; trim then delete)
# =========================================================================== #
def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def pull_and_trim(
    data: EpisodeData,
    puller: Puller,
    *,
    start_s: float,
    max_seconds: float,
    workdir: Path,
) -> Path:
    """Pull the episode mp4 and trim to ``[start_s, start_s+max_seconds]``.

    The full mp4 is deleted right after trimming (only the small clip remains).
    Returns the local path to the trimmed clip.
    """
    vrel = data.video["path"]
    workdir.mkdir(parents=True, exist_ok=True)
    clip = workdir / f"{data.slug}_src_clip.mp4"
    if clip.exists():
        clip.unlink()

    full = puller.pull(vrel)  # cached scp/local copy
    # Trim with re-encode (accurate seek, small output). -ss before -i = fast seek.
    cmd = [
        "ffmpeg", "-y", "-ss", f"{start_s:.3f}", "-i", str(full),
        "-t", f"{max_seconds:.3f}",
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", str(clip),
    ]
    r = _run(cmd)
    if r.returncode != 0 or not clip.exists():
        raise RuntimeError(f"ffmpeg trim failed for {data.slug}: {r.stderr[-800:]}")

    # Delete the big full mp4 (keep only small ones <5MB, e.g. egome/egtea).
    # NEVER delete an in-place local_root/nfs source -- only our own scp'd temp copy.
    try:
        if puller.owns(full) and full.exists() and full.stat().st_size > 5_000_000:
            full.unlink()
    except OSError:
        pass
    return clip


# =========================================================================== #
# 5. DRAWING (PIL only) + ENCODE
# =========================================================================== #
def _load_font(size: int):
    from PIL import ImageFont

    for cand in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ):
        if Path(cand).exists():
            try:
                return ImageFont.truetype(cand, size)
            except Exception:  # noqa
                pass
    return ImageFont.load_default()


def _wrap(text: str, font, max_w: int, draw) -> list[str]:
    words = text.split()
    lines, cur = [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if draw.textlength(trial, font=font) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def draw_frame(
    img,
    *,
    gaze_px: tuple[float, float] | None,
    captions: list[tuple[str, str, tuple[int, int, int]]],
    t: float,
    slug: str,
):
    """Draw the gaze crosshair + a caption band onto a PIL image (in place)."""
    from PIL import ImageDraw

    draw = ImageDraw.Draw(img, "RGBA")
    W, H = img.size

    # --- gaze crosshair + dot ---
    if gaze_px is not None:
        x, y = gaze_px
        if -50 <= x <= W + 50 and -50 <= y <= H + 50:
            r = max(6, W // 90)
            draw.ellipse([x - r, y - r, x + r, y + r], outline=GAZE_COLOR, width=3)
            draw.line([x - r * 2, y, x + r * 2, y], fill=GAZE_COLOR, width=2)
            draw.line([x, y - r * 2, x, y + r * 2], fill=GAZE_COLOR, width=2)
            draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=GAZE_COLOR)

    # --- caption band (bottom) ---
    font_sz = max(12, W // 45)
    font = _load_font(font_sz)
    lh = font_sz + 4
    pad = 6
    rendered: list[tuple[str, tuple[int, int, int]]] = []
    for name, text, color in captions:
        label = f"{name}: {text}" if text else f"{name}: -"
        for ln in _wrap(label, font, W - 2 * pad, draw):
            rendered.append((ln, color))
    # time/HUD line at the top-left
    hud = f"{slug}  t={t:6.2f}s"
    hud_h = lh + 2 * pad
    draw.rectangle([0, 0, draw.textlength(hud, font=font) + 2 * pad, hud_h], fill=(0, 0, 0, 150))
    draw.text((pad, pad), hud, fill=(255, 255, 255), font=font)

    if rendered:
        band_h = len(rendered) * lh + 2 * pad
        draw.rectangle([0, H - band_h, W, H], fill=(0, 0, 0, 160))
        yy = H - band_h + pad
        for ln, color in rendered:
            draw.text((pad, yy), ln, fill=color, font=font)
            yy += lh
    return img


# =========================================================================== #
# 6. TOP-LEVEL RENDER
# =========================================================================== #
@dataclass
class RenderResult:
    out_path: str
    duration_s: float
    width: int
    height: int
    frames: int
    gaze_visible_frames: int
    channels_shown: int
    notes: list[str] = field(default_factory=list)


def render_overlay(
    slug: str,
    episode_id: str,
    puller: Puller,
    out_path: str | Path,
    *,
    max_seconds: float = 20.0,
    start_s: float | None = None,
    side_by_side_gt: str | Path | None = None,
    sample_extra: dict[str, Any] | None = None,
    max_height: int = 720,
    out_fps: float | None = None,
    workdir: str | Path | None = None,
    reuse_bundle: bool = True,
) -> RenderResult:
    """Render a gaze + annotation overlay clip for one episode.

    Pulls the mp4, trims to ``[start_s, start_s+max_seconds]`` (auto-picks a
    window with annotation activity when ``start_s`` is None), projects gaze
    per-frame, draws gaze + captions, and encodes a <=``max_height`` mp4 to
    ``out_path``. When ``side_by_side_gt`` is given, hstacks the overlay with
    that reference video.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    workdir = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix=f"overlay_{slug}_"))
    workdir.mkdir(parents=True, exist_ok=True)

    data = build_episode_data(slug, episode_id, puller, sample_extra=sample_extra, reuse=reuse_bundle)
    fps = float(data.video.get("fps") or 30.0)
    vid_dur = float(data.video.get("duration_s") or max_seconds)

    # Reconcile clocks once.
    gaze_times = reconciled_gaze_times(data)
    annos = reconciled_annotations(data)
    # Parallel validity mask: invalid/dropout rows (placeholders, blinks, tracking
    # loss) must never be selected for drawing -- otherwise the dot snaps to the
    # (0,0)/(0.5,0.5) placeholder or shows stale gaze. Rows without an explicit
    # 'valid' key default to valid (already-2D pixel datasets carry no flag).
    gaze_valid = [bool(r.get("valid", True)) for r in data.gaze_rows]

    # Pick a window with annotation activity if not given.
    if start_s is None:
        start_s = _auto_window(annos, vid_dur, max_seconds)
    span = min(max_seconds, max(0.5, vid_dur - start_s))

    # Pull + trim the source mp4 to the window.
    src_clip = pull_and_trim(
        data, puller, start_s=start_s, max_seconds=span, workdir=workdir
    )

    # Decode the trimmed clip to PNG frames (downscaled to max_height).
    frames_dir = workdir / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True)
    src_meta = _probe(src_clip)
    src_w, src_h = src_meta["width"], src_meta["height"]
    scale = min(1.0, max_height / src_h) if src_h else 1.0
    out_w = int(round(src_w * scale)) // 2 * 2
    out_h = int(round(src_h * scale)) // 2 * 2
    decode_fps = out_fps or fps
    vf = f"fps={decode_fps},scale={out_w}:{out_h}"
    r = _run(["ffmpeg", "-y", "-i", str(src_clip), "-vf", vf,
              str(frames_dir / "f_%06d.png")])
    if r.returncode != 0:
        raise RuntimeError(f"frame decode failed: {r.stderr[-800:]}")
    frame_files = sorted(frames_dir.glob("f_*.png"))
    if not frame_files:
        raise RuntimeError("no frames decoded")

    # Build projection context (loads calib ONCE).
    ctx = build_projection_context(data, puller)
    px_scale = out_w / src_w  # gaze projected in source px -> downscaled frame px

    # Channel colours.
    chan_colors = {ch["name"]: CHANNEL_COLORS[i % len(CHANNEL_COLORS)]
                   for i, ch in enumerate(annos)}

    from PIL import Image

    gaze_visible = 0
    channels_seen: set[str] = set()
    n_frames = len(frame_files)
    for fi, fpath in enumerate(frame_files):
        # frame fi of the trimmed clip -> absolute video time
        t_video = start_s + frame_time(fi, decode_fps)
        # nearest VALID gaze sample (allow up to 1 gaze period of slack); invalid
        # rows are excluded so dropouts blank the dot instead of snapping/staling.
        gi = nearest_gaze_index(
            gaze_times, t_video,
            max_dt=max(0.2, 2.0 / max(1.0, _gaze_hz(gaze_times))),
            valid_mask=gaze_valid,
        )
        gaze_px = None
        if gi is not None:
            proj = project_one(data.gaze_rows[gi], ctx)
            if proj is not None:
                gaze_px = (proj[0] * px_scale, proj[1] * px_scale)
                gaze_visible += 1
        # active annotations per channel
        captions: list[tuple[str, str, tuple[int, int, int]]] = []
        for ch in annos:
            act = active_segments(ch["segments"], t_video, kind=ch["kind"])
            if act:
                channels_seen.add(ch["name"])
                texts = [a.get("text") or "" for a in act]
                txt = " | ".join(t for t in texts if t)[:240]
                captions.append((ch["name"], txt, chan_colors[ch["name"]]))
            else:
                captions.append((ch["name"], "", chan_colors[ch["name"]]))
        img = Image.open(fpath).convert("RGB")
        draw_frame(img, gaze_px=gaze_px, captions=captions, t=t_video, slug=slug)
        img.save(fpath)

    # Encode overlay frames -> mp4.
    overlay_mp4 = workdir / f"{slug}_overlay.mp4"
    r = _run(["ffmpeg", "-y", "-framerate", f"{decode_fps}",
              "-i", str(frames_dir / "f_%06d.png"),
              "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
              "-pix_fmt", "yuv420p", str(overlay_mp4)])
    if r.returncode != 0:
        raise RuntimeError(f"overlay encode failed: {r.stderr[-800:]}")

    final = overlay_mp4
    notes: list[str] = list(ctx.notes)
    if side_by_side_gt is not None:
        final = _hstack(overlay_mp4, Path(side_by_side_gt), out_path, decode_fps)
        notes.append(f"hstacked with GT reference {side_by_side_gt}")
    else:
        shutil.copy2(overlay_mp4, out_path)
        final = out_path

    fin_meta = _probe(out_path)
    return RenderResult(
        out_path=str(out_path),
        duration_s=fin_meta.get("duration") or span,
        width=fin_meta["width"],
        height=fin_meta["height"],
        frames=n_frames,
        gaze_visible_frames=gaze_visible,
        channels_shown=len(channels_seen),
        notes=notes + [f"window=[{start_s:.2f},{start_s+span:.2f}]s fps={decode_fps}"],
    )


def _auto_window(annos: list[dict[str, Any]], vid_dur: float, max_seconds: float) -> float:
    """Pick a start time so the clip overlaps annotation activity."""
    starts = []
    for ch in annos:
        for s in ch["segments"]:
            v = s.get("start_s") if s.get("start_s") is not None else s.get("point_s")
            if v is not None and 0 <= v <= vid_dur:
                starts.append(v)
    if not starts:
        return 0.0
    first = min(starts)
    # Start a couple seconds before the first activity, clamped to video.
    start = max(0.0, min(first - 2.0, max(0.0, vid_dur - max_seconds)))
    return start


def _gaze_hz(gaze_times: list[float | None]) -> float:
    vals = [t for t in gaze_times if t is not None]
    if len(vals) < 2:
        return 30.0
    span = vals[-1] - vals[0]
    return (len(vals) / span) if span > 0 else 30.0


def _probe(path: Path) -> dict[str, Any]:
    r = _run(["ffprobe", "-v", "error", "-select_streams", "v:0",
              "-show_entries", "stream=width,height,duration",
              "-show_entries", "format=duration", "-of", "json", str(path)])
    data = json.loads(r.stdout or "{}")
    st = (data.get("streams") or [{}])[0]
    fmt = data.get("format", {})
    dur = st.get("duration") or fmt.get("duration")
    return {
        "width": int(st.get("width") or 0),
        "height": int(st.get("height") or 0),
        "duration": float(dur) if dur not in (None, "N/A") else None,
    }


def _hstack(left: Path, right: Path, out_path: Path, fps: float) -> Path:
    """Horizontally stack two videos (scaled to equal height) via ffmpeg."""
    lm = _probe(left)
    h = lm["height"] or 720
    vf = (
        f"[0:v]scale=-2:{h}[l];[1:v]scale=-2:{h}[r];[l][r]hstack=inputs=2"
    )
    r = _run(["ffmpeg", "-y", "-i", str(left), "-i", str(right),
              "-filter_complex", vf, "-c:v", "libx264", "-preset", "veryfast",
              "-crf", "20", "-pix_fmt", "yuv420p", str(out_path)])
    if r.returncode != 0:
        raise RuntimeError(f"hstack failed: {r.stderr[-800:]}")
    return out_path
