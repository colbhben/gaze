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


def discover_manifest_episodes(
    manifest_path: str | Path,
    dataset: str | None = None,
    episodes: set[str] | None = None,
) -> list[RawEpisode]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    result: list[RawEpisode] = []
    for dataset_name, doc in sorted((manifest.get("datasets") or {}).items()):
        if dataset and dataset_name != dataset:
            continue
        root_value = doc.get("root")
        if not root_value:
            continue
        dataset_root = Path(root_value)
        files_by_sequence: dict[str, list[dict[str, Any]]] = {}
        for file_doc in doc.get("files") or []:
            rel = Path(file_doc["path"])
            sequence_id = rel.parts[0] if len(rel.parts) > 1 else "dataset"
            files_by_sequence.setdefault(sequence_id, []).append(file_doc)
        for sequence_id, seq_doc in sorted((doc.get("sequences") or {}).items()):
            if episodes and sequence_id not in episodes:
                continue
            files = select_episode_files(dataset_root, files_by_sequence.get(sequence_id, []))
            if not files:
                continue
            duration = seq_doc.get("duration_s") or infer_duration(files)
            result.append(
                RawEpisode(
                    dataset=dataset_name,
                    episode_id=sequence_id,
                    root=dataset_root / sequence_id if (dataset_root / sequence_id).exists() else dataset_root,
                    duration_s=float(duration or 0.0),
                    files=files,
                    metadata={"raw_manifest": str(manifest_path), "raw_manifest_sequence": seq_doc},
                )
            )
    return result


def select_episode_files(dataset_root: Path, files: list[dict[str, Any]]) -> dict[str, Path]:
    selected: dict[str, Path] = {}
    modality_priority = {
        "gaze": ("table", "json"),
        "annotations": ("table", "json"),
        "depth": ("table", "json"),
        "video": ("video",),
    }
    for target_key, accepted_formats in modality_priority.items():
        modality = "annotation" if target_key == "annotations" else target_key
        candidates = [
            file_doc
            for file_doc in files
            if file_doc.get("modality") == modality and file_doc.get("format") in accepted_formats
        ]
        if not candidates:
            continue
        candidates.sort(key=lambda item: (0 if item.get("format") == accepted_formats[0] else 1, item.get("size_bytes") or 0))
        selected[target_key] = dataset_root / candidates[-1]["path"]
    return selected


def infer_duration(files: dict[str, Path]) -> float:
    max_time = 0.0
    for key in ("gaze", "annotations", "annotation_intervals", "depth", "video_frames"):
        path = files.get(key)
        if not path or not path.exists():
            continue
        rows = read_raw_records(path)
        for row in rows:
            normalized = normalize_time_fields(row)
            for column in ("time_s", "start_s", "end_s"):
                if normalized.get(column) is not None:
                    max_time = max(max_time, float(normalized[column]))
    return max_time


def rectify_dataset(
    raw_root: str | Path,
    canonical_root: str | Path,
    config: RectifyConfig | None = None,
    dataset: str | None = None,
    episodes: set[str] | None = None,
    raw_manifest: str | Path | None = None,
) -> list[dict[str, Any]]:
    cfg = config or default_config()
    output_root = Path(canonical_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    raw_episodes = discover_raw_episodes(raw_root, dataset=dataset, episodes=episodes)
    if raw_manifest:
        known = {(episode.dataset, episode.episode_id) for episode in raw_episodes}
        for episode in discover_manifest_episodes(raw_manifest, dataset=dataset, episodes=episodes):
            if (episode.dataset, episode.episode_id) not in known:
                raw_episodes.append(episode)
    for raw_episode in raw_episodes:
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
        gaze_rows = rectify_gaze(read_raw_records(raw.files["gaze"]), timeline)
        output_files["gaze"] = str(write_table(gaze_rows, episode_root / "gaze.parquet").relative_to(episode_root))

    annotation_path = raw.files.get("annotations") or raw.files.get("annotation_intervals")
    if annotation_path and annotation_path.exists():
        source_annotations = read_raw_records(annotation_path)
        sampled, intervals = rectify_annotations(source_annotations, timeline)
        output_files["annotations"] = str(write_table(sampled, episode_root / "annotations.parquet").relative_to(episode_root))
        if intervals:
            output_files["annotation_intervals"] = str(
                write_table(intervals, episode_root / "annotation_intervals.parquet").relative_to(episode_root)
            )

    if cfg.depth.enabled and raw.files.get("depth") and raw.files["depth"].exists():
        depth_rows = rectify_depth(read_raw_records(raw.files["depth"]), timeline)
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
    rows = normalize_gaze_rows(rows)
    columns = [column for column in ("x_norm", "y_norm", "x_px", "y_px") if any(row.get(column) is not None for row in rows)]
    return [
        {
            "time_s": point["time_s"],
            **{column: interpolate_numeric(rows, point["time_s"], column) for column in columns},
        }
        for point in timeline
    ]


def rectify_depth(rows: list[dict[str, Any]], timeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = normalize_depth_rows(rows)
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
    rows = normalize_annotation_rows(rows)
    if not rows:
        return ([{"time_s": point["time_s"], "label": None, "text": None} for point in timeline], [])
    interval_like = any(row.get("start_s") is not None or row.get("end_s") is not None for row in rows)
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


def read_raw_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if source.suffix == ".json":
        data = json.loads(source.read_text(encoding="utf-8"))
        return json_records(data)
    return read_table(source)


def json_records(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        list_values = [value for value in data.values() if isinstance(value, list)]
        dict_lists = [[item for item in value if isinstance(item, dict)] for value in list_values]
        dict_lists = [value for value in dict_lists if value]
        if dict_lists:
            return max(dict_lists, key=len)
        return [data]
    return []


def normalize_gaze_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        item = normalize_time_fields(row)
        x_norm = first_present(row, "x_norm", "yolo_x_norm", "gaze_x_norm", "norm_x")
        y_norm = first_present(row, "y_norm", "yolo_y_norm", "gaze_y_norm", "norm_y")
        x_px = first_present(row, "x_px", "gaze_x_px", "gaze_x", "x", "u")
        y_px = first_present(row, "y_px", "gaze_y_px", "gaze_y", "y", "v")
        if x_norm is None and y_norm is None and x_px is not None and y_px is not None and 0 <= float(x_px) <= 1 and 0 <= float(y_px) <= 1:
            x_norm, y_norm = x_px, y_px
            x_px, y_px = None, None
        item.update({"x_norm": x_norm, "y_norm": y_norm, "x_px": x_px, "y_px": y_px})
        normalized.append(item)
    return normalized


def normalize_depth_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        item = normalize_time_fields(row)
        item["depth_m"] = first_present(row, "depth_m", "depth", "z", "distance_m")
        normalized.append(item)
    return normalized


def normalize_annotation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        item = normalize_time_fields(row)
        item["label"] = first_present(row, "label", "action", "action_name", "class", "verb", "noun")
        text = first_present(row, "text", "narration", "description", "commentary", "activity", "summary")
        if text is None and row.get("verb") is not None and row.get("noun") is not None:
            text = f"{row.get('verb')} {row.get('noun')}"
        item["text"] = text
        normalized.append(item)
    return normalized


def normalize_time_fields(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "time_s": first_time(row, "time_s", "timestamp_s", "timestamp_ns", "timestamp_ms", "timestamp", "time", "t", "video_time_s"),
        "start_s": first_time(row, "start_s", "start_time_s", "start_time_ns", "start_time_ms", "start_time", "start", "start_timestamp"),
        "end_s": first_time(row, "end_s", "end_time_s", "end_time_ns", "end_time_ms", "end_time", "end", "end_timestamp"),
    }


def first_present(row: dict[str, Any], *columns: str) -> Any:
    lowered = {key.lower(): value for key, value in row.items()}
    for column in columns:
        if column in row and row[column] is not None:
            return row[column]
        value = lowered.get(column.lower())
        if value is not None:
            return value
    return None


def first_time(row: dict[str, Any], *columns: str) -> float | None:
    for column in columns:
        value = first_present(row, column)
        if value is not None:
            return normalize_time_value(value, column)
    return None


def normalize_time_value(value: Any, column: str) -> float:
    number = float(value)
    lowered = column.lower()
    if lowered.endswith("_ns") or "nanosecond" in lowered or abs(number) > 1e12:
        return number / 1_000_000_000.0
    if lowered.endswith("_ms") or "millisecond" in lowered:
        return number / 1_000.0
    if abs(number) > 1e6:
        return number / 1_000.0
    return number


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
        if path.suffix in {".json", ".jsonl", ".csv", ".parquet"}:
            rows = read_raw_records(path)
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
