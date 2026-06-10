"""Recipe-driven extraction harness.

Reads a `recipes/<slug>.json` curation recipe and extracts ONE episode's
egocentric video metadata, gaze table, and annotation channels into a
canonical per-episode bundle — without modifying any source data.

Design (per the project decisions):
  - The mac is the processing host (has ffprobe/ffmpeg/numpy/pandas/
    projectaria_tools); the remote NFS is read-only. Source files are pulled
    via the configured `Puller` (ssh/scp by default).
  - Gaze is preserved NATIVE; the gaze table records coordinate space + a
    projection recipe. Aria/psi projection to 2D is applied only when asked.
  - Annotations are kept as RAW segments per named channel, plus timing stats.
  - Emit policy (video + gaze + >=1 annotation) is enforced by `extract_episode`;
    a missing optional piece warns and is recorded, it does not raise.

This module is import-light (stdlib only at import time); heavy deps
(numpy/pandas/projectaria_tools/ffprobe) are imported lazily inside the
readers that need them, so the recipe loader/validator path stays portable.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable


# --------------------------------------------------------------------------- #
# Recipe loading
# --------------------------------------------------------------------------- #
RECIPES_DIR_DEFAULT = Path(__file__).resolve().parents[2] / "recipes"


def load_recipe(slug: str, recipes_dir: str | Path | None = None) -> dict[str, Any]:
    base = Path(recipes_dir) if recipes_dir else RECIPES_DIR_DEFAULT
    defaults_path = base / "_defaults.json"
    defaults = json.loads(defaults_path.read_text(encoding="utf-8")) if defaults_path.exists() else {}
    recipe = json.loads((base / f"{slug}.json").read_text(encoding="utf-8"))
    merged = dict(defaults)
    merged.update(recipe)  # recipe overrides defaults at the top level
    merged["_defaults"] = defaults
    return merged


# --------------------------------------------------------------------------- #
# Pulling source files (remote read-only -> local temp)
# --------------------------------------------------------------------------- #
class Puller:
    """Pulls source files/dirs from the data host to a local working dir.

    Default implementation reads from a `/nfs` mount reachable over ssh on a
    remote host, copying via scp. When `local_root` is set (e.g. the mount is
    visible locally), files are read in place / copied locally instead.
    """

    def __init__(
        self,
        ssh_host: str | None = "sumedhso-L40S",
        remote_root: str = "/nfs/colbhben/gaze/unprocessed",
        local_root: str | Path | None = None,
        workdir: str | Path | None = None,
    ) -> None:
        self.ssh_host = ssh_host
        self.remote_root = remote_root.rstrip("/")
        self.local_root = Path(local_root) if local_root else None
        self.workdir = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="gaze_curate_"))
        self.workdir.mkdir(parents=True, exist_ok=True)

    def remote_path(self, rel: str) -> str:
        return f"{self.remote_root}/{rel.lstrip('/')}"

    def exists(self, rel: str) -> bool:
        if self.local_root:
            return (self.local_root / rel).exists()
        result = subprocess.run(
            ["ssh", self.ssh_host, f"test -e {_shq(self.remote_path(rel))}"],
            capture_output=True,
        )
        return result.returncode == 0

    def listdir(self, rel: str) -> list[str]:
        if self.local_root:
            d = self.local_root / rel
            return sorted(p.name for p in d.iterdir()) if d.exists() else []
        result = subprocess.run(
            ["ssh", self.ssh_host, f"ls -1 {_shq(self.remote_path(rel))} 2>/dev/null"],
            capture_output=True, text=True,
        )
        return [line for line in result.stdout.splitlines() if line]

    def glob(self, rel_glob: str) -> list[str]:
        """Glob relative to the source root; returns matching relative paths."""
        if self.local_root:
            base = self.local_root
            return sorted(str(p.relative_to(base)) for p in base.glob(rel_glob))
        # remote: use bash globbing via ls
        result = subprocess.run(
            ["ssh", self.ssh_host, f"ls -1d {_shq(self.remote_path(rel_glob), glob=True)} 2>/dev/null"],
            capture_output=True, text=True,
        )
        out = []
        for line in result.stdout.splitlines():
            if line.startswith(self.remote_root + "/"):
                out.append(line[len(self.remote_root) + 1:])
        return out

    def pull(self, rel: str, *, binary: bool = True) -> Path:
        """Copy a single source file local; return the local path (cached)."""
        dest = self.workdir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.stat().st_size > 0:
            return dest
        if self.local_root:
            shutil.copy2(self.local_root / rel, dest)
        else:
            subprocess.run(
                ["scp", "-q", f"{self.ssh_host}:{self.remote_path(rel)}", str(dest)],
                check=True,
            )
        return dest

    def read_text(self, rel: str, *, max_bytes: int | None = None) -> str:
        """Read a (small) source text file without caching a full copy."""
        if self.local_root:
            data = (self.local_root / rel).read_bytes()
            return (data[:max_bytes] if max_bytes else data).decode("utf-8", "replace")
        cmd = f"cat {_shq(self.remote_path(rel))}"
        if max_bytes:
            cmd = f"head -c {int(max_bytes)} {_shq(self.remote_path(rel))}"
        result = subprocess.run(["ssh", self.ssh_host, cmd], capture_output=True, text=True)
        return result.stdout


def _shq(path: str, *, glob: bool = False) -> str:
    """Shell-quote a remote path. When glob=True, leave * and ? unquoted."""
    if glob:
        # quote everything except glob metachars by splitting on them
        return path  # globs are pre-trusted (built from recipe templates)
    return "'" + path.replace("'", "'\\''") + "'"


# --------------------------------------------------------------------------- #
# Episode bundle (the extraction output)
# --------------------------------------------------------------------------- #
@dataclass
class GazeTable:
    coordinate_space: str
    columns: list[str]
    rows: list[dict[str, Any]]
    hz: float | None = None
    sample_count: int = 0
    duration_s: float | None = None
    valid_fraction: float | None = None
    projection: dict[str, Any] | None = None
    source_path: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class AnnotationChannel:
    name: str
    kind: str
    segments: list[dict[str, Any]]  # raw: {start_s|None, end_s|None, point_s|None, text, extras{}}
    segment_count: int = 0
    coverage_s: float | None = None
    coverage_fraction: float | None = None
    mean_rate_hz: float | None = None
    first_start_s: float | None = None
    last_end_s: float | None = None
    source_path: str | None = None


@dataclass
class VideoMeta:
    path: str
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    duration_s: float | None = None
    frame_count: int | None = None
    codec: str | None = None
    source: str = "ffprobe"  # or "shipped"


@dataclass
class EpisodeBundle:
    dataset: str
    episode_id: str
    video: VideoMeta | None = None
    gaze: GazeTable | None = None
    annotations: list[AnnotationChannel] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    emitted: bool = False
    emit_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "episode_id": self.episode_id,
            "video": asdict(self.video) if self.video else None,
            "gaze": _gaze_summary(self.gaze) if self.gaze else None,
            "annotations": [_anno_summary(a) for a in self.annotations],
            "warnings": self.warnings,
            "emitted": self.emitted,
            "emit_reason": self.emit_reason,
        }


def _gaze_summary(g: GazeTable) -> dict[str, Any]:
    d = asdict(g)
    d["rows"] = len(g.rows)  # don't inline the full table in the summary
    return d


def _anno_summary(a: AnnotationChannel) -> dict[str, Any]:
    d = asdict(a)
    d["segments"] = len(a.segments)
    return d


# --------------------------------------------------------------------------- #
# ffprobe video metadata
# --------------------------------------------------------------------------- #
def ffprobe_video(path: Path) -> VideoMeta:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,codec_name,duration",
        "-show_entries", "format=duration",
        "-of", "json", str(path),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(out.stdout or "{}")
    stream = (data.get("streams") or [{}])[0]
    fmt = data.get("format", {})
    fps = _parse_rate(stream.get("avg_frame_rate")) or _parse_rate(stream.get("r_frame_rate"))
    duration = _to_float(stream.get("duration")) or _to_float(fmt.get("duration"))
    nb = _to_int(stream.get("nb_frames"))
    return VideoMeta(
        path=str(path),
        fps=fps,
        width=_to_int(stream.get("width")),
        height=_to_int(stream.get("height")),
        duration_s=duration,
        frame_count=nb,
        codec=stream.get("codec_name"),
        source="ffprobe",
    )


def _parse_rate(value: str | None) -> float | None:
    if not value or value in ("0/0", "N/A"):
        return None
    if "/" in value:
        n, d = value.split("/", 1)
        d = float(d)
        return None if d == 0 else round(float(n) / d, 6)
    return _to_float(value)


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Time unit normalization
# --------------------------------------------------------------------------- #
def to_seconds(value: Any, units: str, fps: float | None = None) -> float | None:
    """Convert a raw time value to seconds per the recipe's declared units."""
    if value is None or value == "":
        return None
    if units == "s":
        return _to_float(value)
    if units == "ms":
        return _to_float(value) / 1_000.0
    if units == "us":
        return _to_float(value) / 1_000_000.0
    if units == "ns":
        return _to_float(value) / 1_000_000_000.0
    if units == "filetime_100ns":
        return _to_float(value) / 1e7
    if units == "frame_index":
        if not fps:
            raise ValueError("frame_index units require fps")
        return _to_float(value) / fps
    if units == "hms_decimal":
        return _parse_hms_decimal(str(value))
    if units == "timecode_hmsf":
        if not fps:
            raise ValueError("timecode_hmsf units require fps")
        return _parse_timecode_hmsf(str(value), fps)
    raise ValueError(f"unknown time units: {units}")


def _parse_hms_decimal(text: str) -> float | None:
    """HH:MM:SS.mmm decimal-seconds string -> seconds."""
    text = text.strip()
    if not text:
        return None
    parts = text.split(":")
    parts = [float(p) for p in parts]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    h, m, s = parts[-3], parts[-2], parts[-1]
    return h * 3600 + m * 60 + s


def _parse_timecode_hmsf(text: str, fps: float) -> float | None:
    """HH:MM:SS:FF frame-timecode -> seconds (FF = frame within second)."""
    text = text.strip()
    if not text:
        return None
    h, m, s, f = (int(x) for x in text.split(":"))
    return h * 3600 + m * 60 + s + f / fps


# --------------------------------------------------------------------------- #
# Annotation timing stats
# --------------------------------------------------------------------------- #
def annotation_stats(channel_name: str, kind: str, segments: list[dict[str, Any]], video_duration_s: float | None) -> AnnotationChannel:
    starts = [s["start_s"] for s in segments if s.get("start_s") is not None]
    ends = [s["end_s"] for s in segments if s.get("end_s") is not None]
    points = [s["point_s"] for s in segments if s.get("point_s") is not None]
    first = min(starts + points) if (starts or points) else None
    last = max(ends + points) if (ends or points) else None
    coverage = None
    if kind == "interval" and segments:
        coverage = _interval_union(
            [(s["start_s"], s["end_s"]) for s in segments
             if s.get("start_s") is not None and s.get("end_s") is not None]
        )
    coverage_fraction = (coverage / video_duration_s) if (coverage is not None and video_duration_s) else None
    span = (last - first) if (first is not None and last is not None and last > first) else None
    mean_rate = (len(segments) / span) if span else None
    return AnnotationChannel(
        name=channel_name,
        kind=kind,
        segments=segments,
        segment_count=len(segments),
        coverage_s=coverage,
        coverage_fraction=coverage_fraction,
        mean_rate_hz=round(mean_rate, 6) if mean_rate else None,
        first_start_s=first,
        last_end_s=last,
    )


def _interval_union(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    total = 0.0
    cur_start, cur_end = ordered[0]
    for start, end in ordered[1:]:
        if start > cur_end:
            total += cur_end - cur_start
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    total += cur_end - cur_start
    return round(total, 6)


# --------------------------------------------------------------------------- #
# Text composition (compose / strip_prefix)
# --------------------------------------------------------------------------- #
def compose_text(row: dict[str, Any], text_spec: dict[str, Any]) -> str | None:
    if "primary" in text_spec:
        value = _dotted_get(row, text_spec["primary"])
        text = None if value is None else str(value)
    else:
        parts = [_dotted_get(row, key) for key in text_spec.get("compose", [])]
        parts = [str(p) for p in parts if p not in (None, "", "none", "None")]
        text = (text_spec.get("compose_sep", " ").join(parts)) or None
    if text and text_spec.get("strip_prefix_regex"):
        text = re.sub(text_spec["strip_prefix_regex"], "", text).strip()
    return text


def _dotted_get(row: dict[str, Any], key: str) -> Any:
    """Get a possibly-dotted key (e.g. 'attributes.Long form description')."""
    if key in row:
        return row[key]
    cur: Any = row
    for part in key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


# NOTE: the per-reader implementations (gaze csv/npy/begaze/whitespace, annotation
# json_by_key/pandas_pickle, and the Aria/psi projection) are provided in
# `curate_readers.py` and wired through `extract_episode` there. This module holds
# the stable contract: recipe loading, pulling, dataclasses, ffprobe, time/units,
# stats, and text composition.
