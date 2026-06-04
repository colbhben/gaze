from __future__ import annotations

from pathlib import Path
import shutil
import struct
import subprocess
from typing import Any
import os


def find_ffmpeg() -> str | None:
    configured = os.environ.get("GAZE_FFMPEG")
    if configured and Path(configured).exists():
        return configured
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def video_duration_s(path: str | Path) -> float | None:
    source = Path(path)
    if not source.exists():
        return None
    return mp4_mvhd_duration_s(source)


def transcode_video(source: Path, target: Path, cfg: Any) -> dict[str, Any]:
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        shutil.copyfile(source, target)
        return {"transcoded": False, "reason": "ffmpeg not found"}

    target.parent.mkdir(parents=True, exist_ok=True)
    filters = [f"fps={cfg.video.fps}"]
    if cfg.video.width and cfg.video.height:
        if cfg.video.resize_mode == "stretch":
            filters.append(f"scale={cfg.video.width}:{cfg.video.height}")
        elif cfg.video.resize_mode == "crop":
            filters.append(
                f"scale={cfg.video.width}:{cfg.video.height}:force_original_aspect_ratio=increase,"
                f"crop={cfg.video.width}:{cfg.video.height}"
            )
        elif cfg.video.resize_mode == "pad":
            filters.append(
                f"scale={cfg.video.width}:{cfg.video.height}:force_original_aspect_ratio=decrease,"
                f"pad={cfg.video.width}:{cfg.video.height}:(ow-iw)/2:(oh-ih)/2"
            )
        else:
            filters.append(f"scale={cfg.video.width}:{cfg.video.height}:force_original_aspect_ratio=decrease")
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-vf",
        ",".join(filters),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        str(target),
    ]
    try:
        completed = subprocess.run(command, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        shutil.copyfile(source, target)
        return {
            "transcoded": False,
            "reason": "ffmpeg failed",
            "command": command,
            "returncode": exc.returncode,
            "stderr": exc.stderr,
        }
    return {
        "transcoded": True,
        "command": command,
        "returncode": completed.returncode,
    }


def mp4_mvhd_duration_s(path: Path) -> float | None:
    size = path.stat().st_size
    with path.open("rb") as handle:
        return _scan_boxes_for_mvhd(handle, 0, size)


def _scan_boxes_for_mvhd(handle, start: int, end: int) -> float | None:
    handle.seek(start)
    while handle.tell() + 8 <= end:
        box_start = handle.tell()
        header = handle.read(8)
        if len(header) < 8:
            return None
        size, box_type_raw = struct.unpack(">I4s", header)
        box_type = box_type_raw.decode("latin1")
        header_size = 8
        if size == 1:
            extended = handle.read(8)
            if len(extended) < 8:
                return None
            size = struct.unpack(">Q", extended)[0]
            header_size = 16
        elif size == 0:
            size = end - box_start
        if size < header_size:
            return None
        payload = box_start + header_size
        box_end = box_start + size
        if box_type == "mvhd":
            return _read_mvhd_duration(handle, payload)
        if box_type == "moov":
            found = _scan_boxes_for_mvhd(handle, payload, box_end)
            if found is not None:
                return found
        handle.seek(box_end)
    return None


def _read_mvhd_duration(handle, offset: int) -> float | None:
    handle.seek(offset)
    version_raw = handle.read(1)
    if not version_raw:
        return None
    version = version_raw[0]
    handle.read(3)
    if version == 1:
        handle.read(16)
        timescale_raw = handle.read(4)
        duration_raw = handle.read(8)
        if len(timescale_raw) < 4 or len(duration_raw) < 8:
            return None
        timescale = struct.unpack(">I", timescale_raw)[0]
        duration = struct.unpack(">Q", duration_raw)[0]
    else:
        handle.read(8)
        timescale_raw = handle.read(4)
        duration_raw = handle.read(4)
        if len(timescale_raw) < 4 or len(duration_raw) < 4:
            return None
        timescale = struct.unpack(">I", timescale_raw)[0]
        duration = struct.unpack(">I", duration_raw)[0]
    if timescale == 0:
        return None
    return duration / timescale
