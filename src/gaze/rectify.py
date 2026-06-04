from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .config import RectifyConfig, default_config
from .table import read_table, write_table
from .video import transcode_video


@dataclass
class RawEpisode:
    dataset: str
    episode_id: str
    root: Path
    duration_s: float
    files: dict[str, Path]
    metadata: dict[str, Any]


def discover_raw_episodes(raw_root: str | Path, dataset: str | None = None, episodes: set[str] | None = None) -> list[RawEpisode]:
    root = Path(raw_root)
    candidates = sorted(root.glob("**/episode.json"))
    result: list[RawEpisode] = []
    for episode_file in candidates:
        raw = json.loads(episode_file.read_text(encoding="utf-8"))
        dataset_name = raw.get("dataset") or episode_file.parent.parent.name
        episode_id = raw.get("episode_id") or episode_file.parent.name
        if dataset and dataset_name != dataset:
            continue
        if episodes and episode_id not in episodes:
            continue
        files = {}
        for key, value in raw.get("files", {}).items():
            path = Path(value)
            files[key] = path if path.is_absolute() else episode_file.parent / path
        duration = raw.get("duration_s")
        if duration is None:
            duration = infer_duration(files)
        result.append(
            RawEpisode(
                dataset=dataset_name,
                episode_id=episode_id,
                root=episode_file.parent,
                duration_s=float(duration),
                files=files,
                metadata=raw,
            )
        )
    return result


def infer_duration(files: dict[str, Path]) -> float:
    max_time = 0.0
    for key in ("gaze", "annotations", "annotation_intervals", "depth", "video_frames"):
        path = files.get(key)
        if not path or not path.exists():
            continue
        rows = read_table(path)
        for row in rows:
            for column in ("time_s", "start_s", "end_s"):
                if row.get(column) is not None:
                    max_time = max(max_time, float(row[column]))
    return max_time


def rectify_dataset(
    raw_root: str | Path,
    canonical_root: str | Path,
    config: RectifyConfig | None = None,
    dataset: str | None = None,
    episodes: set[str] | None = None,
) -> list[dict[str, Any]]:
    cfg = config or default_config()
    output_root = Path(canonical_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for raw_episode in discover_raw_episodes(raw_root, dataset=dataset, episodes=episodes):
        manifest_rows.append(rectify_episode(raw_episode, output_root, cfg))
    write_table(manifest_rows, output_root / "manifest.parquet")
    return manifest_rows


def rectify_episode(raw: RawEpisode, canonical_root: Path, cfg: RectifyConfig) -> dict[str, Any]:
    episode_root = canonical_root / "episodes" / raw.dataset / raw.episode_id
    episode_root.mkdir(parents=True, exist_ok=True)
    timeline = make_timeline(raw.duration_s, cfg.target_hz)
    source_fingerprint = fingerprint_episode(raw)
    output_files: dict[str, str] = {}

    output_files["timeline"] = str(write_table(timeline, episode_root / "timeline.parquet").relative_to(episode_root))

    video_file = raw.files.get("video")
    video_info = None
    if video_file and video_file.exists():
        target_video = episode_root / f"video.{cfg.video.format}"
        if video_file.resolve() != target_video.resolve():
            video_info = transcode_video(video_file, target_video, cfg)
        output_files["video"] = target_video.name

    if raw.files.get("gaze") and raw.files["gaze"].exists():
        gaze_rows = rectify_gaze(read_table(raw.files["gaze"]), timeline)
        output_files["gaze"] = str(write_table(gaze_rows, episode_root / "gaze.parquet").relative_to(episode_root))

    annotation_path = raw.files.get("annotations") or raw.files.get("annotation_intervals")
    if annotation_path and annotation_path.exists():
        source_annotations = read_table(annotation_path)
        sampled, intervals = rectify_annotations(source_annotations, timeline)
        output_files["annotations"] = str(write_table(sampled, episode_root / "annotations.parquet").relative_to(episode_root))
        if intervals:
            output_files["annotation_intervals"] = str(
                write_table(intervals, episode_root / "annotation_intervals.parquet").relative_to(episode_root)
            )

    if cfg.depth.enabled and raw.files.get("depth") and raw.files["depth"].exists():
        depth_rows = rectify_depth(read_table(raw.files["depth"]), timeline)
        output_files["depth"] = str(write_table(depth_rows, episode_root / "depth.parquet").relative_to(episode_root))

    episode_doc = {
        "dataset": raw.dataset,
        "episode_id": raw.episode_id,
        "duration_s": raw.duration_s,
        "config": cfg.to_dict(),
        "source": {
            "root": str(raw.root),
            "files": {key: str(value) for key, value in raw.files.items()},
            "fingerprint": source_fingerprint,
        },
        "video_processing": video_info,
        "files": output_files,
        "modalities": sorted(k for k in output_files if k not in {"timeline", "annotation_intervals"}),
    }
    (episode_root / "episode.json").write_text(json.dumps(episode_doc, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "dataset": raw.dataset,
        "episode_id": raw.episode_id,
        "duration_s": raw.duration_s,
        "episode_path": str(episode_root.relative_to(canonical_root)),
        "modalities": ",".join(episode_doc["modalities"]),
    }


def make_timeline(duration_s: float, hz: float) -> list[dict[str, Any]]:
    if duration_s < 0:
        raise ValueError("duration_s must be non-negative")
    step = 1.0 / hz
    count = int(math.floor((duration_s + 1e-9) / step)) + 1
    return [{"frame_index": index, "time_s": round(index * step, 9)} for index in range(count)]


def rectify_gaze(rows: list[dict[str, Any]], timeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
    columns = [column for column in ("x_norm", "y_norm", "x_px", "y_px") if any(row.get(column) is not None for row in rows)]
    return [
        {
            "time_s": point["time_s"],
            **{column: interpolate_numeric(rows, point["time_s"], column) for column in columns},
        }
        for point in timeline
    ]


def rectify_depth(rows: list[dict[str, Any]], timeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "time_s": point["time_s"],
            "depth_m": interpolate_numeric(rows, point["time_s"], "depth_m"),
        }
        for point in timeline
    ]


def rectify_annotations(
    rows: list[dict[str, Any]], timeline: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not rows:
        return ([{"time_s": point["time_s"], "label": None, "text": None} for point in timeline], [])
    interval_like = any("start_s" in row or "end_s" in row for row in rows)
    if interval_like:
        intervals = normalize_intervals(rows)
        final_end = max((row["end_s"] for row in intervals), default=0.0)
        sampled = []
        for point in timeline:
            active = [
                row
                for row in intervals
                if row["start_s"] <= point["time_s"] < row["end_s"]
                or (math.isclose(point["time_s"], final_end) and math.isclose(point["time_s"], row["end_s"]))
            ]
            sampled.append(
                {
                    "time_s": point["time_s"],
                    "label": "|".join(str(row.get("label", "")) for row in active) or None,
                    "text": "|".join(str(row.get("text", "")) for row in active) or None,
                }
            )
        return sampled, intervals
    sampled = []
    for point in timeline:
        nearest = nearest_row(rows, point["time_s"])
        sampled.append(
            {
                "time_s": point["time_s"],
                "label": nearest.get("label") if nearest else None,
                "text": nearest.get("text") if nearest else None,
            }
        )
    return sampled, []


def normalize_intervals(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        normalized.append(
            {
                "start_s": float(row.get("start_s", row.get("time_s", 0.0))),
                "end_s": float(row.get("end_s", row.get("time_s", 0.0))),
                "label": row.get("label"),
                "text": row.get("text"),
            }
        )
    return normalized


def interpolate_numeric(rows: list[dict[str, Any]], time_s: float, column: str) -> float | None:
    points = sorted(
        ((float(row["time_s"]), row.get(column)) for row in rows if row.get("time_s") is not None and row.get(column) is not None),
        key=lambda item: item[0],
    )
    if not points:
        return None
    if time_s < points[0][0] or time_s > points[-1][0]:
        return None
    for index, (point_time, value) in enumerate(points):
        if math.isclose(time_s, point_time, abs_tol=1e-9):
            return float(value)
        if point_time > time_s:
            before_time, before_value = points[index - 1]
            span = point_time - before_time
            if span == 0:
                return float(before_value)
            ratio = (time_s - before_time) / span
            return float(before_value) + ratio * (float(value) - float(before_value))
    return float(points[-1][1])


def nearest_row(rows: list[dict[str, Any]], time_s: float) -> dict[str, Any] | None:
    timed = [row for row in rows if row.get("time_s") is not None]
    if not timed:
        return None
    return min(timed, key=lambda row: abs(float(row["time_s"]) - time_s))


def fingerprint_episode(raw: RawEpisode) -> dict[str, Any]:
    modalities = {}
    for modality, path in raw.files.items():
        if not path.exists() or path.is_dir():
            continue
        info: dict[str, Any] = {"path": str(path), "sha1": sha1_file(path), "bytes": path.stat().st_size}
        if path.suffix in {".jsonl", ".csv", ".parquet"}:
            rows = read_table(path)
            info.update(table_fingerprint(rows))
        modalities[modality] = info
    return {"duration_s": raw.duration_s, "modalities": modalities}


def table_fingerprint(rows: list[dict[str, Any]]) -> dict[str, Any]:
    columns = sorted({column for row in rows for column in row})
    times = [float(row["time_s"]) for row in rows if row.get("time_s") is not None]
    starts = [float(row["start_s"]) for row in rows if row.get("start_s") is not None]
    ends = [float(row["end_s"]) for row in rows if row.get("end_s") is not None]
    samples = []
    if rows:
        indexes = sorted({0, len(rows) // 2, len(rows) - 1})
        samples = [rows[index] for index in indexes]
    return {
        "rows": len(rows),
        "columns": columns,
        "time_min_s": min(times or starts or [0.0]),
        "time_max_s": max(times or ends or [0.0]),
        "samples": samples,
    }


def sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
