from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import hashlib
import json
import shutil
import subprocess
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Any

from .datasets import DatasetCatalog, load_catalog
from .download import plan_downloads, target_path_for_asset


VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".vrs"}
TABLE_SUFFIXES = {".csv", ".jsonl", ".parquet"}
JSON_SUFFIXES = {".json"}
ARCHIVE_SUFFIXES = {".zip", ".tar", ".tgz", ".gz", ".bz2", ".xz"}
PARTIAL_SUFFIXES = {".part"}


DATASET_FORMAT_HINTS: dict[str, dict[str, Any]] = {
    "aea": {
        "provider": "Aria Everyday Activities",
        "expected_raw": "Project Aria VRS recordings with MPS eye-gaze CSV and activity/narration annotations.",
        "video": {"typical_fps": "provider metadata or VRS stream metadata", "encoding": "Project Aria VRS/RGB stream"},
        "gaze": {"typical_fps": "provider metadata or inferred from MPS timestamps", "encoding": "MPS eye-gaze CSV; normalized/pixel columns are derived during rectification when available"},
        "annotations": {"format": "CSV annotation/narration files from provider manifest"},
    },
    "hot3d": {
        "provider": "HOT3D Aria",
        "expected_raw": "Aria VRS recordings, MPS eye gaze, point clouds, and hand/object annotations.",
        "video": {"typical_fps": "provider metadata or VRS stream metadata", "encoding": "Project Aria VRS/RGB stream"},
        "gaze": {"typical_fps": "provider metadata or inferred from timestamps", "encoding": "MPS eye-gaze tables"},
        "annotations": {"format": "HOT3D hand/object/take annotations from provider files"},
    },
    "nymeria": {
        "provider": "Nymeria",
        "expected_raw": "Preview RGB videos, head/observer VRS recordings, metadata JSON, motion files, and narration CSVs.",
        "video": {"typical_fps": "ffprobe/VRS metadata when available", "encoding": "MP4 preview videos and VRS sensor recordings"},
        "gaze": {"typical_fps": "provider metadata or inferred from timestamps", "encoding": "head/eye related streams when exported from provider files"},
        "annotations": {"format": "narration CSV files including atomic action, motion narration, and activity summarization"},
    },
    "adt": {
        "provider": "Aria Digital Twin",
        "expected_raw": "Project Aria RGB/VRS recordings with MPS eye-gaze and SLAM files plus scene ground truth, segmentation, depth, and synthetic assets.",
        "video": {"typical_fps": "provider metadata or VRS stream metadata", "encoding": "Preview RGB MP4 and main VRS recordings"},
        "gaze": {"typical_fps": "inferred from MPS eye-gaze CSV timestamps", "encoding": "MPS eye-gaze CSV"},
        "annotations": {"format": "main ground-truth zip contents from provider manifest"},
        "depth": {"format": "provider depth zip contents"},
    },
    "holoassist": {
        "provider": "HoloAssist",
        "expected_raw": "HoloLens video, eye-gaze, depth, and JSON annotations.",
        "video": {"typical_fps": "ffprobe metadata when available", "encoding": "MP4 video clips"},
        "gaze": {"typical_fps": "inferred from eye timestamp columns", "encoding": "HoloLens eye-gaze tables/rays"},
        "annotations": {"format": "provider JSON train/validation annotations"},
    },
    "egtea": {
        "provider": "EGTEA Gaze+",
        "expected_raw": "Video clips, per-frame gaze files, and action annotation text/CSV files.",
        "video": {"typical_fps": "ffprobe metadata when available", "encoding": "RGB video clips"},
        "gaze": {"typical_fps": "inferred from gaze rows or frame ids", "encoding": "pixel or normalized gaze records depending on source file"},
        "annotations": {"format": "action interval annotations"},
    },
    "ego-exo4d": {
        "provider": "Ego-Exo4D",
        "expected_raw": "Downloaded takes, eye-gaze tables, and benchmark annotation JSON files from Ego4D manifests.",
        "video": {"typical_fps": "ffprobe metadata when available", "encoding": "ego-view videos under takes"},
        "gaze": {"typical_fps": "inferred from take_eye_gaze timestamps", "encoding": "take eye-gaze tables"},
        "annotations": {"format": "benchmark JSON annotations"},
    },
}


CANONICAL_OUTPUT_SPEC: dict[str, Any] = {
    "layout": {
        "manifest": "manifest.parquet, or manifest.parquet.jsonl when parquet dependencies are unavailable",
        "episode": "episodes/{dataset}/{episode_id}/episode.json",
        "timeline": "episodes/{dataset}/{episode_id}/timeline.parquet",
        "video": "episodes/{dataset}/{episode_id}/video.{format}",
        "gaze": "episodes/{dataset}/{episode_id}/gaze.parquet",
        "annotations": "episodes/{dataset}/{episode_id}/annotations.parquet",
        "annotation_intervals": "episodes/{dataset}/{episode_id}/annotation_intervals.parquet when interval annotations exist",
        "depth": "episodes/{dataset}/{episode_id}/depth.parquet when depth is enabled and present",
    },
    "timeline_schema": {"frame_index": "int, zero-based", "time_s": "float seconds on the configured target_hz grid"},
    "gaze_schema": {
        "time_s": "float seconds aligned to timeline",
        "x_norm": "optional float in [0, 1]",
        "y_norm": "optional float in [0, 1]",
        "x_px": "optional float pixel x",
        "y_px": "optional float pixel y",
    },
    "annotation_schema": {"time_s": "float seconds aligned to timeline", "label": "optional string", "text": "optional string"},
    "annotation_interval_schema": {"start_s": "float seconds", "end_s": "float seconds", "label": "optional string", "text": "optional string"},
    "depth_schema": {"time_s": "float seconds aligned to timeline", "depth_m": "optional non-negative float meters"},
}


@dataclass
class ExpectedAssetStatus:
    dataset: str
    sequence_id: str
    asset_key: str
    modality: str
    path: str
    complete: bool
    size_bytes: int | None
    expected_size_bytes: int | None
    sha1: str | None
    expected_sha1: str | None
    reason: str | None = None


def wait_for_downloads_and_write_manifest(
    raw_root: str | Path,
    output: str | Path,
    copy_out: str | Path | None = None,
    repo_root: str | Path = ".",
    datasets: set[str] | None = None,
    modalities: set[str] | None = None,
    sequences: set[str] | None = None,
    poll_seconds: float = 60.0,
    stable_checks: int = 2,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    consecutive_complete = 0
    last_status: dict[str, Any] | None = None
    while True:
        catalog = load_catalog(repo_root)
        statuses = expected_asset_statuses(raw_root, catalog, datasets=datasets, modalities=modalities, sequences=sequences)
        complete = Path(raw_root).exists() and all(status.complete for status in statuses)
        partials = find_partial_downloads(raw_root)
        if complete and not partials:
            consecutive_complete += 1
        else:
            consecutive_complete = 0
        last_status = {
            "expected_assets": len(statuses),
            "complete_assets": sum(1 for status in statuses if status.complete),
            "partial_downloads": [str(path) for path in partials],
            "stable_checks": consecutive_complete,
            "required_stable_checks": stable_checks,
        }
        if consecutive_complete >= stable_checks:
            manifest = inspect_raw_root(raw_root, repo_root=repo_root, datasets=datasets, statuses=statuses)
            manifest["watcher"] = {
                "host": "sumedhso-L40S",
                "poll_seconds": poll_seconds,
                "stable_checks": stable_checks,
                "status": last_status,
            }
            written = write_manifest(manifest, output)
            if copy_out:
                copy_manifest(written, copy_out)
            return manifest
        if timeout_seconds is not None and time.monotonic() - started > timeout_seconds:
            raise TimeoutError(f"downloads did not finish before timeout: {json.dumps(last_status, sort_keys=True)}")
        time.sleep(poll_seconds)


def inspect_raw_root(
    raw_root: str | Path,
    repo_root: str | Path = ".",
    datasets: set[str] | None = None,
    modalities: set[str] | None = None,
    sequences: set[str] | None = None,
    statuses: list[ExpectedAssetStatus] | None = None,
) -> dict[str, Any]:
    root = Path(raw_root)
    catalog = load_catalog(repo_root)
    asset_statuses = statuses if statuses is not None else expected_asset_statuses(
        root,
        catalog,
        datasets=datasets,
        modalities=modalities,
        sequences=sequences,
    )
    dataset_dirs = sorted(path for path in root.iterdir() if path.is_dir()) if root.exists() else []
    selected = {item.lower() for item in datasets or set()}
    datasets_doc: dict[str, Any] = {}
    for dataset_dir in dataset_dirs:
        dataset = dataset_dir.name
        if selected and dataset.lower() not in selected:
            continue
        datasets_doc[dataset] = inspect_dataset_dir(dataset_dir, dataset)
    for status in asset_statuses:
        datasets_doc.setdefault(status.dataset, inspect_missing_dataset(status.dataset))
    return {
        "version": 1,
        "created_at_unix": int(time.time()),
        "raw_root": str(root),
        "datasets": datasets_doc,
        "expected_assets": [asdict(status) for status in asset_statuses],
        "canonical_output": CANONICAL_OUTPUT_SPEC,
    }


def write_manifest(manifest: dict[str, Any], output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path


def copy_manifest(source: str | Path, destination: str | Path) -> Path:
    dest = Path(destination)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    return dest


def expected_asset_statuses(
    raw_root: str | Path,
    catalog: DatasetCatalog,
    datasets: set[str] | None = None,
    modalities: set[str] | None = None,
    sequences: set[str] | None = None,
) -> list[ExpectedAssetStatus]:
    root = Path(raw_root)
    statuses = []
    for asset in plan_downloads(catalog, datasets=datasets, modalities=modalities, sequences=sequences):
        target = target_path_for_asset(root, asset)
        size = target.stat().st_size if target.exists() and target.is_file() else None
        reason = None
        complete = target.exists()
        if not complete:
            reason = "missing"
        elif asset.size_bytes is not None and target.is_file() and size != asset.size_bytes:
            complete = False
            reason = f"size mismatch: {size} != {asset.size_bytes}"
        elif target.with_suffix(target.suffix + ".part").exists():
            complete = False
            reason = "partial file exists"
        actual_sha1 = None
        if complete and asset.sha1 and target.is_file():
            actual_sha1 = sha1_file(target)
            complete = actual_sha1 == asset.sha1
            if not complete:
                reason = "sha1 mismatch"
        statuses.append(
            ExpectedAssetStatus(
                dataset=asset.dataset,
                sequence_id=asset.sequence_id,
                asset_key=asset.asset_key,
                modality=asset.modality,
                path=str(target),
                complete=complete,
                size_bytes=size,
                expected_size_bytes=asset.size_bytes,
                sha1=actual_sha1,
                expected_sha1=asset.sha1,
                reason=reason,
            )
        )
    return statuses


def inspect_missing_dataset(dataset: str) -> dict[str, Any]:
    return {
        "root": None,
        "format_hints": DATASET_FORMAT_HINTS.get(dataset, {}),
        "files": [],
        "sequences": {},
        "observed_modalities": [],
        "ready_for_rectification": False,
        "warnings": ["dataset directory was not observed under raw root"],
    }


def inspect_dataset_dir(dataset_dir: Path, dataset: str) -> dict[str, Any]:
    files = []
    sequence_docs: dict[str, dict[str, Any]] = {}
    warnings = []
    for path in sorted(item for item in dataset_dir.rglob("*") if item.is_file()):
        if path.name.startswith("."):
            continue
        rel = path.relative_to(dataset_dir)
        summary = summarize_file(path, dataset_dir)
        files.append(summary)
        sequence_id = rel.parts[0] if len(rel.parts) > 1 else "dataset"
        seq = sequence_docs.setdefault(sequence_id, {"files": 0, "modalities": set(), "formats": set(), "duration_s": None})
        seq["files"] += 1
        seq["modalities"].add(summary["modality"])
        seq["formats"].add(summary["format"])
        duration = summary.get("duration_s") or summary.get("table", {}).get("time_max_s")
        if duration is not None:
            seq["duration_s"] = max(seq["duration_s"] or 0.0, float(duration))
        if summary["format"] == "archive":
            warnings.append(f"archive observed and not extracted: {rel}")
    observed_modalities = sorted({item["modality"] for item in files})
    sequences = {
        seq_id: {
            **doc,
            "modalities": sorted(doc["modalities"]),
            "formats": sorted(doc["formats"]),
        }
        for seq_id, doc in sequence_docs.items()
    }
    return {
        "root": str(dataset_dir),
        "format_hints": DATASET_FORMAT_HINTS.get(dataset, {}),
        "files": files,
        "sequences": sequences,
        "observed_modalities": observed_modalities,
        "ready_for_rectification": any(can_file_drive_rectification(item) for item in files),
        "warnings": sorted(set(warnings)),
    }


def summarize_file(path: Path, root: Path) -> dict[str, Any]:
    suffix = normalized_suffix(path)
    rel = path.relative_to(root)
    summary: dict[str, Any] = {
        "path": str(rel),
        "name": path.name,
        "size_bytes": path.stat().st_size,
        "suffix": suffix,
        "format": classify_format(path),
        "modality": infer_modality(path),
    }
    if suffix in TABLE_SUFFIXES:
        summary["table"] = inspect_table(path)
    elif suffix in JSON_SUFFIXES:
        summary["json"] = inspect_json(path)
    elif suffix in VIDEO_SUFFIXES:
        summary.update(inspect_video(path))
    elif suffix in ARCHIVE_SUFFIXES:
        summary["archive"] = inspect_archive(path)
    return summary


def inspect_table(path: Path) -> dict[str, Any]:
    try:
        rows, total_rows, columns = sample_table(path)
    except Exception as exc:
        return {"error": str(exc)}
    time_values = time_values_from_rows(rows)
    return {
        "rows": total_rows,
        "sampled_rows": len(rows),
        "columns": columns,
        "time_columns": [column for column in columns if is_time_column(column)],
        "gaze_encoding": infer_gaze_encoding(columns),
        "annotation_format": infer_annotation_format(columns),
        "fps": estimate_frequency_hz(time_values),
        "time_min_s": min(time_values) if time_values else None,
        "time_max_s": max(time_values) if time_values else None,
        "samples": sample_rows(rows),
    }


def sample_table(path: Path, max_rows: int = 1000) -> tuple[list[dict[str, Any]], int | None, list[str]]:
    if path.suffix == ".csv":
        rows = []
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = list(reader.fieldnames or [])
            for index, row in enumerate(reader):
                if index >= max_rows:
                    break
                rows.append({key: coerce_cell(value) for key, value in row.items()})
        return rows, count_csv_rows(path), columns
    if path.suffix == ".jsonl":
        rows = []
        total = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    total += 1
                    if len(rows) < max_rows:
                        item = json.loads(line)
                        if isinstance(item, dict):
                            rows.append(item)
        columns = sorted({column for row in rows for column in row})
        return rows, total, columns
    if path.suffix == ".parquet":
        return sample_parquet(path, max_rows=max_rows)
    return [], None, []


def sample_parquet(path: Path, max_rows: int) -> tuple[list[dict[str, Any]], int | None, list[str]]:
    try:
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(path)
        columns = parquet_file.schema.names
        table = parquet_file.read_row_group(0, columns=columns) if parquet_file.num_row_groups else None
        rows = table.slice(0, max_rows).to_pylist() if table is not None else []
        total = parquet_file.metadata.num_rows if parquet_file.metadata else None
        return rows, total, columns
    except Exception:
        import pandas as pd

        frame = pd.read_parquet(path)
        rows = json.loads(frame.head(max_rows).to_json(orient="records"))
        return rows, len(frame), list(frame.columns)


def count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def coerce_cell(value: str) -> Any:
    if value == "":
        return None
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        if "." not in value and "e" not in lowered:
            return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def inspect_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"error": str(exc)}
    if isinstance(data, dict):
        keys = sorted(data)[:100]
        return {"type": "object", "keys": keys, "annotation_format": infer_annotation_format(keys)}
    if isinstance(data, list):
        keys = sorted({key for item in data[:100] if isinstance(item, dict) for key in item})[:100]
        return {"type": "array", "items_sampled": min(len(data), 100), "keys": keys, "annotation_format": infer_annotation_format(keys)}
    return {"type": type(data).__name__}


def inspect_video(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".vrs":
        return {"video_container": "vrs", "duration_s": None, "fps": None}
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return {"video_container": path.suffix.lower().lstrip("."), "duration_s": None, "fps": None, "warning": "ffprobe not found"}
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,width,height,duration,codec_name",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        return {"video_container": path.suffix.lower().lstrip("."), "error": completed.stderr[-500:]}
    data = json.loads(completed.stdout or "{}")
    stream = (data.get("streams") or [{}])[0]
    return {
        "video_container": path.suffix.lower().lstrip("."),
        "codec": stream.get("codec_name"),
        "width": stream.get("width"),
        "height": stream.get("height"),
        "fps": parse_frame_rate(stream.get("avg_frame_rate")),
        "duration_s": float(stream["duration"]) if stream.get("duration") not in {None, "N/A"} else None,
    }


def inspect_archive(path: Path, max_members: int = 200) -> dict[str, Any]:
    try:
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                names = archive.namelist()[:max_members]
        elif tarfile.is_tarfile(path):
            with tarfile.open(path) as archive:
                names = archive.getnames()[:max_members]
        else:
            return {"type": "compressed", "members_sampled": 0, "modalities": []}
    except Exception as exc:
        return {"error": str(exc)}
    return {
        "members_sampled": len(names),
        "sample_members": names[:25],
        "modalities": sorted({infer_modality(Path(name)) for name in names}),
        "formats": sorted({classify_format(Path(name)) for name in names}),
    }


def can_file_drive_rectification(summary: dict[str, Any]) -> bool:
    if summary["format"] in {"table", "json", "video"}:
        return True
    if summary.get("archive", {}).get("modalities"):
        return False
    return False


def infer_modality(path: Path) -> str:
    text = "/".join(path.parts).lower()
    if any(token in text for token in ("gaze", "eye", "eyes", "etdata")):
        return "gaze"
    if any(token in text for token in ("annot", "narration", "action", "label", "ground_truth")):
        return "annotation"
    if any(token in text for token in ("depth", "point_cloud", "semidense")):
        return "depth"
    if any(token in text for token in ("pose", "hand", "motion", "mvnx", "slam_trajectories")):
        return "pose"
    if normalized_suffix(path) in VIDEO_SUFFIXES or any(token in text for token in ("video", "rgb", "preview", "takes")):
        return "video"
    return "other"


def classify_format(path: Path) -> str:
    suffix = normalized_suffix(path)
    if suffix in VIDEO_SUFFIXES:
        return "video"
    if suffix in TABLE_SUFFIXES:
        return "table"
    if suffix in JSON_SUFFIXES:
        return "json"
    if suffix in ARCHIVE_SUFFIXES:
        return "archive"
    return "binary"


def normalized_suffix(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".tar.gz") or name.endswith(".tgz"):
        return ".tgz"
    return path.suffix.lower()


def infer_gaze_encoding(columns: list[str]) -> str | None:
    lowered = {column.lower() for column in columns}
    if {"x_norm", "y_norm"} <= lowered:
        return "normalized_2d"
    if {"x_px", "y_px"} <= lowered or {"x", "y"} <= lowered or {"gaze_x", "gaze_y"} <= lowered:
        return "pixel_2d"
    if any("yaw" in column or "pitch" in column for column in lowered):
        return "angular"
    if any("origin" in column or "direction" in column for column in lowered):
        return "ray"
    return None


def infer_annotation_format(columns: list[str]) -> str | None:
    lowered = {column.lower() for column in columns}
    if lowered & {"start_s", "end_s", "start_time", "end_time", "start_frame", "end_frame"}:
        return "interval"
    if lowered & {"label", "text", "narration", "action", "verb", "noun"}:
        return "event_or_label"
    return None


def is_time_column(column: str) -> bool:
    lowered = column.lower()
    return (
        lowered in {"time_s", "timestamp", "timestamp_s", "timestamp_ms", "timestamp_ns", "time", "t"}
        or lowered.endswith("_time")
        or lowered.endswith("_time_s")
        or lowered.endswith("_time_ms")
        or lowered.endswith("_time_ns")
        or lowered.endswith("_timestamp")
    )


def time_values_from_rows(rows: list[dict[str, Any]]) -> list[float]:
    values = []
    for row in rows:
        for column in ("time_s", "timestamp_s", "timestamp_ms", "timestamp_ns", "timestamp", "time", "start_s", "end_s"):
            value = row.get(column)
            if value is not None:
                values.append(normalize_time_value(value, column))
                break
    return sorted(values)


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


def estimate_frequency_hz(times: list[float]) -> float | None:
    unique = sorted(set(round(value, 9) for value in times))
    deltas = [b - a for a, b in zip(unique, unique[1:]) if b > a]
    if not deltas:
        return None
    deltas.sort()
    median = deltas[len(deltas) // 2]
    if median <= 0:
        return None
    return round(1.0 / median, 6)


def sample_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    indexes = sorted({0, len(rows) // 2, len(rows) - 1})
    return [rows[index] for index in indexes]


def parse_frame_rate(value: str | None) -> float | None:
    if not value or value == "0/0":
        return None
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        denom = float(denominator)
        return None if denom == 0 else round(float(numerator) / denom, 6)
    return float(value)


def find_partial_downloads(raw_root: str | Path) -> list[Path]:
    root = Path(raw_root)
    if not root.exists():
        return []
    partials = []
    for path in root.rglob("*"):
        if path.is_file() and (path.suffix in PARTIAL_SUFFIXES or ".ranges" in path.parts):
            partials.append(path)
    return sorted(partials)


def sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
