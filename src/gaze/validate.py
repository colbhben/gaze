from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any

from .config import RectifyConfig
from .rectify import RawEpisode, make_timeline, read_raw_records, rectify_annotations, rectify_depth, rectify_gaze
from .table import read_table


@dataclass
class CheckResult:
    name: str
    ok: bool
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def validate_canonical_root(canonical_root: str | Path, raw_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(canonical_root)
    reports = []
    for episode_file in sorted(root.glob("episodes/*/*/episode.json")):
        reports.append(validate_episode(episode_file, raw_root=Path(raw_root) if raw_root else None))
    ok = all(report["ok"] for report in reports)
    report = {"ok": ok, "episodes": reports}
    (root / "alignment_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def validate_episode(episode_file: str | Path, raw_root: Path | None = None) -> dict[str, Any]:
    episode_path = Path(episode_file)
    episode_root = episode_path.parent
    doc = json.loads(episode_path.read_text(encoding="utf-8"))
    cfg = _config_from_doc(doc)
    raw = _raw_episode_from_doc(doc, raw_root=raw_root)
    checks = [
        check_cross_modal(doc, episode_root, cfg),
        check_video(doc, episode_root, raw, cfg),
        check_gaze(doc, episode_root, raw, cfg),
        check_annotations(doc, episode_root, raw, cfg),
        check_depth(doc, episode_root, raw, cfg),
    ]
    result = {
        "dataset": doc["dataset"],
        "episode_id": doc["episode_id"],
        "ok": all(check.ok for check in checks),
        "checks": [check.__dict__ for check in checks],
    }
    (episode_root / "alignment_report.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def _config_from_doc(doc: dict[str, Any]) -> RectifyConfig:
    from .config import load_config

    config_file_value = doc.get("config_file")
    config_file = Path(config_file_value) if config_file_value else None
    if config_file and config_file.exists() and config_file.is_file():
        return load_config(config_file)
    return RectifyConfig(
        profile_name=doc["config"].get("profile_name", "default-10hz"),
        target_hz=float(doc["config"].get("target_hz", 10.0)),
        validation=RectifyConfig().validation.__class__(**doc["config"].get("validation", {})),
    )


def _raw_episode_from_doc(doc: dict[str, Any], raw_root: Path | None = None) -> RawEpisode:
    source = doc.get("source", {})
    files = {}
    for key, value in source.get("files", {}).items():
        path = Path(value)
        if raw_root and not path.exists():
            candidate = raw_root / doc["dataset"] / doc["episode_id"] / path.name
            if candidate.exists():
                path = candidate
        files[key] = path
    root = Path(source.get("root") or (raw_root or "."))
    return RawEpisode(
        dataset=doc["dataset"],
        episode_id=doc["episode_id"],
        root=root,
        duration_s=float(doc["duration_s"]),
        files=files,
        metadata={},
    )


def table_path(episode_root: Path, doc: dict[str, Any], key: str) -> Path | None:
    value = doc.get("files", {}).get(key)
    if not value:
        return None
    return episode_root / value


def check_cross_modal(doc: dict[str, Any], episode_root: Path, cfg: RectifyConfig) -> CheckResult:
    result = CheckResult("cross_modal", True)
    timeline_path = table_path(episode_root, doc, "timeline")
    if not timeline_path:
        result.ok = False
        result.errors.append("missing timeline")
        return result
    timeline = read_table(timeline_path)
    expected = make_timeline(float(doc["duration_s"]), cfg.target_hz)
    result.metrics["timeline_rows"] = len(timeline)
    result.metrics["expected_rows"] = len(expected)
    if len(timeline) != len(expected):
        result.ok = False
        result.errors.append("timeline sample count does not match configured grid")
    for actual, exp in zip(timeline, expected):
        if abs(float(actual["time_s"]) - float(exp["time_s"])) > cfg.validation.time_tolerance_s:
            result.ok = False
            result.errors.append("timeline timestamps diverge from configured grid")
            break
    seen = set()
    for modality in doc.get("modalities", []):
        if modality in seen:
            result.ok = False
            result.errors.append(f"duplicate modality {modality}")
        seen.add(modality)
    return result


def check_video(doc: dict[str, Any], episode_root: Path, raw: RawEpisode, cfg: RectifyConfig) -> CheckResult:
    result = CheckResult("video", True)
    if "video" not in doc.get("files", {}):
        result.warnings.append("no canonical video")
        return result
    timeline_path = table_path(episode_root, doc, "timeline")
    timeline = read_table(timeline_path) if timeline_path else []
    expected = make_timeline(float(doc["duration_s"]), cfg.target_hz)
    if len(timeline) != len(expected):
        result.ok = False
        result.errors.append("video timeline does not match target frame count")
    if timeline:
        duration = float(timeline[-1]["time_s"])
        drift = abs(duration - expected[-1]["time_s"])
        result.metrics["frame_time_drift_s"] = drift
        if drift > cfg.validation.frame_time_tolerance_s:
            result.ok = False
            result.errors.append("video frame timestamp drift exceeds tolerance")
    source_video = raw.files.get("video")
    canonical_video = episode_root / doc["files"]["video"]
    if source_video and source_video.exists() and canonical_video.exists():
        result.metrics["source_video_bytes"] = source_video.stat().st_size
        result.metrics["canonical_video_bytes"] = canonical_video.stat().st_size
    return result


def check_gaze(doc: dict[str, Any], episode_root: Path, raw: RawEpisode, cfg: RectifyConfig) -> CheckResult:
    result = CheckResult("gaze", True)
    raw_path = raw.files.get("gaze")
    canonical_path = table_path(episode_root, doc, "gaze")
    if not raw_path or not raw_path.exists():
        result.warnings.append("no raw gaze")
        return result
    if not canonical_path:
        result.ok = False
        result.errors.append("raw gaze exists but canonical gaze is missing")
        return result
    timeline = make_timeline(float(doc["duration_s"]), cfg.target_hz)
    expected = rectify_gaze(read_raw_records(raw_path), timeline)
    actual = read_table(canonical_path)
    compare_rows("gaze", expected, actual, cfg.validation.numeric_tolerance, result)
    for row in actual:
        for column in ("x_norm", "y_norm"):
            value = row.get(column)
            if value is not None and not (0.0 <= float(value) <= 1.0):
                result.ok = False
                result.errors.append(f"{column} out of normalized bounds")
                return result
    return result


def check_annotations(doc: dict[str, Any], episode_root: Path, raw: RawEpisode, cfg: RectifyConfig) -> CheckResult:
    result = CheckResult("annotations", True)
    raw_path = raw.files.get("annotations") or raw.files.get("annotation_intervals")
    canonical_path = table_path(episode_root, doc, "annotations")
    if not raw_path or not raw_path.exists():
        result.warnings.append("no raw annotations")
        return result
    if not canonical_path:
        result.ok = False
        result.errors.append("raw annotations exist but canonical annotations are missing")
        return result
    expected, expected_intervals = rectify_annotations(read_raw_records(raw_path), make_timeline(float(doc["duration_s"]), cfg.target_hz))
    actual = read_table(canonical_path)
    compare_rows("annotations", expected, actual, 0.0, result, exact_columns={"label", "text"})
    interval_path = table_path(episode_root, doc, "annotation_intervals")
    if expected_intervals and not interval_path:
        result.ok = False
        result.errors.append("source intervals were not preserved")
    if expected_intervals and interval_path:
        actual_intervals = read_table(interval_path)
        compare_rows("annotation_intervals", expected_intervals, actual_intervals, cfg.validation.numeric_tolerance, result, exact_columns={"label", "text"})
    return result


def check_depth(doc: dict[str, Any], episode_root: Path, raw: RawEpisode, cfg: RectifyConfig) -> CheckResult:
    result = CheckResult("depth", True)
    raw_path = raw.files.get("depth")
    canonical_path = table_path(episode_root, doc, "depth")
    if not raw_path or not raw_path.exists():
        result.warnings.append("no raw depth")
        return result
    if not canonical_path:
        result.ok = False
        result.errors.append("raw depth exists but canonical depth is missing")
        return result
    expected = rectify_depth(read_raw_records(raw_path), make_timeline(float(doc["duration_s"]), cfg.target_hz))
    actual = read_table(canonical_path)
    compare_rows("depth", expected, actual, cfg.validation.numeric_tolerance, result)
    for row in actual:
        value = row.get("depth_m")
        if value is not None and float(value) < 0:
            result.ok = False
            result.errors.append("depth_m must be expressed as non-negative meters")
            break
    return result


def compare_rows(
    label: str,
    expected: list[dict[str, Any]],
    actual: list[dict[str, Any]],
    tolerance: float,
    result: CheckResult,
    exact_columns: set[str] | None = None,
) -> None:
    exact_columns = exact_columns or set()
    result.metrics[f"{label}_expected_rows"] = len(expected)
    result.metrics[f"{label}_actual_rows"] = len(actual)
    if len(expected) != len(actual):
        result.ok = False
        result.errors.append(f"{label} row count mismatch")
        return
    for row_index, (exp, act) in enumerate(zip(expected, actual)):
        for column, exp_value in exp.items():
            act_value = act.get(column)
            if column in exact_columns:
                if _norm_text(exp_value) != _norm_text(act_value):
                    result.ok = False
                    result.errors.append(f"{label}.{column} changed at row {row_index}")
                    return
            elif not values_close(exp_value, act_value, tolerance):
                result.ok = False
                result.errors.append(f"{label}.{column} differs at row {row_index}: expected {exp_value!r}, got {act_value!r}")
                return


def values_close(expected: Any, actual: Any, tolerance: float) -> bool:
    if expected is None and actual is None:
        return True
    if expected is None or actual is None:
        return False
    try:
        exp_float = float(expected)
        act_float = float(actual)
        if math.isnan(exp_float) and math.isnan(act_float):
            return True
        return abs(exp_float - act_float) <= tolerance
    except (TypeError, ValueError):
        return expected == actual


def _norm_text(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
