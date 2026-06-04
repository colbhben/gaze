#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import zipfile

from gaze.video import video_duration_s


DEFAULT_NFS_ROOT = Path("/nfs/colbhben/gaze/unprocessed")


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a small rectification-ready raw subset from the /nfs dataset mount.")
    parser.add_argument("--nfs-root", type=Path, default=DEFAULT_NFS_ROOT)
    parser.add_argument("--output-root", type=Path, default=Path("rectify_subsets/raw"))
    parser.add_argument("--max-duration-s", type=float, default=90.0)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    create_aea(args.nfs_root, args.output_root, args.max_duration_s)
    create_hot3d(args.nfs_root, args.output_root, args.max_duration_s)
    create_egtea(args.nfs_root, args.output_root, args.max_duration_s)
    create_nymeria(args.nfs_root, args.output_root, args.max_duration_s)
    episodes = sorted(str(path.relative_to(args.output_root)) for path in args.output_root.glob("*/*/episode.json"))
    print(json.dumps({"raw_root": str(args.output_root), "episodes": episodes}, indent=2))
    return 0


def create_aea(nfs_root: Path, output_root: Path, max_duration_s: float) -> None:
    for episode in ["loc1_script1_seq1_rec1", "loc1_script1_seq3_rec1"]:
        source = nfs_root / "aea" / episode
        root = output_root / "aea" / episode
        root.mkdir(parents=True, exist_ok=True)
        video = next((source / "video_main_rgb").glob("*.mp4"))
        duration = video_duration_s(video) or max_duration_s
        gaze_rows, gaze_base_s = read_aria_gaze(next((source / "mps_eye_gaze").glob("*.zip")), duration)
        annotations = read_aea_speech(next((source / "annotations").glob("*.zip")), base_s=gaze_base_s, max_duration_s=duration)
        write_csv(gaze_rows, root / "gaze.csv")
        write_csv(annotations, root / "annotations.csv")
        write_episode(
            output_root,
            "aea",
            episode,
            duration,
            {
                "video": video,
                "gaze": root / "gaze.csv",
                "annotations": root / "annotations.csv",
            },
        )


def create_hot3d(nfs_root: Path, output_root: Path, max_duration_s: float) -> None:
    for episode in ["P0001_10a27bf7", "P0001_15c4300c"]:
        source = nfs_root / "hot3d" / episode
        root = output_root / "hot3d" / episode
        root.mkdir(parents=True, exist_ok=True)
        video = next((source / "video_main_rgb").glob("*.mp4"))
        duration = video_duration_s(video) or max_duration_s
        gaze_rows, gaze_base_s = read_aria_gaze(next((source / "mps_eye_gaze").glob("*.zip")), duration)
        annotations = read_hot3d_objects(next((source / "ground_truth").glob("*.zip")), base_s=gaze_base_s, max_duration_s=duration)
        write_csv(gaze_rows, root / "gaze.csv")
        write_csv(annotations, root / "annotations.csv")
        write_episode(
            output_root,
            "hot3d",
            episode,
            duration,
            {
                "video": video,
                "gaze": root / "gaze.csv",
                "annotations": root / "annotations.csv",
            },
        )


def create_egtea(nfs_root: Path, output_root: Path, max_duration_s: float) -> None:
    gaze_zip = nfs_root / "egtea" / "dataset" / "gaze" / "gaze_data.zip"
    annotation_zip = nfs_root / "egtea" / "dataset" / "annotations" / "action_annotation.zip"
    sessions = ["P26-R05-Cheeseburger", "P25-R06-GreekSalad"]
    with zipfile.ZipFile(gaze_zip) as archive:
        gaze_members = {Path(name).stem: name for name in archive.namelist() if name.startswith("gaze_data/gaze_data/") and name.endswith(".txt")}
    labels = read_egtea_labels(annotation_zip)
    for session in sessions:
        member = gaze_members.get(session)
        if not member:
            continue
        episode = session.lower().replace("-", "_")
        root = output_root / "egtea" / episode
        root.mkdir(parents=True, exist_ok=True)
        source_labels = labels.get(session, [])
        base_ms = source_labels[0]["start_ms"] if source_labels else 0.0
        gaze_rows = read_egtea_gaze(gaze_zip, member, max_duration_s)
        annotations = [
            {
                "start_s": round((row["start_ms"] - base_ms) / 1000.0, 6),
                "end_s": round(min((row["end_ms"] - base_ms) / 1000.0, max_duration_s), 6),
                "label": row["label"],
                "text": row["label"],
            }
            for row in source_labels
            if 0.0 <= (row["start_ms"] - base_ms) / 1000.0 <= max_duration_s
        ][:20]
        write_csv(gaze_rows, root / "gaze.csv")
        write_csv(annotations or [{"start_s": 0.0, "end_s": max_duration_s, "label": "egtea", "text": session}], root / "annotations.csv")
        write_episode(
            output_root,
            "egtea",
            episode,
            max_duration(gaze_rows, annotations, max_duration_s),
            {"gaze": root / "gaze.csv", "annotations": root / "annotations.csv"},
        )


def create_nymeria(nfs_root: Path, output_root: Path, max_duration_s: float) -> None:
    for episode in ["20230607_s0_james_johnson_act0_e72nhq", "20230607_s0_james_johnson_act1_7xwm28"]:
        source = nfs_root / "nymeria" / episode
        root = output_root / "nymeria" / episode
        root.mkdir(parents=True, exist_ok=True)
        csv_path = next((source / "narration_atomic_action_csv").glob("*.csv"))
        with csv_path.open(encoding="utf-8", newline="") as handle:
            source_rows = list(csv.DictReader(handle))
        base_s = min((float(row["start_time"]) for row in source_rows), default=0.0)
        annotations = []
        for row in source_rows:
            start_s = float(row["start_time"]) - base_s
            end_s = float(row["end_time"]) - base_s
            if start_s > max_duration_s:
                break
            annotations.append(
                {
                    "start_s": round(start_s, 6),
                    "end_s": round(min(end_s, max_duration_s), 6),
                    "label": "atomic_action",
                    "text": row.get("Describe my atomic actions") or "",
                }
            )
        write_csv(annotations, root / "annotations.csv")
        write_episode(output_root, "nymeria", episode, max_duration([], annotations, max_duration_s), {"annotations": root / "annotations.csv"})


def read_aria_gaze(path: Path, max_duration_s: float) -> tuple[list[dict], float]:
    rows = []
    base_s = None
    with zipfile.ZipFile(path) as archive, archive.open("general_eye_gaze.csv") as handle:
        text = (line.decode("utf-8", errors="replace") for line in handle)
        for row in csv.DictReader(text):
            timestamp_s = float(row["tracking_timestamp_us"]) / 1_000_000.0
            if base_s is None:
                base_s = timestamp_s
            time_s = timestamp_s - base_s
            if time_s > max_duration_s:
                break
            yaw = row.get("yaw_rads_cpf")
            if yaw in (None, ""):
                yaw = (float(row.get("left_yaw_rads_cpf") or 0.0) + float(row.get("right_yaw_rads_cpf") or 0.0)) / 2.0
            pitch = float(row.get("pitch_rads_cpf") or 0.0)
            x_norm, y_norm = aria_yaw_pitch_to_normalized(float(yaw), pitch)
            rows.append({"time_s": round(time_s, 6), "x_norm": x_norm, "y_norm": y_norm})
    return rows, base_s or 0.0


def read_aea_speech(path: Path, base_s: float, max_duration_s: float) -> list[dict]:
    rows = []
    with zipfile.ZipFile(path) as archive, archive.open("speech.csv") as handle:
        text = (line.decode("utf-8", errors="replace") for line in handle)
        for row in csv.DictReader(text):
            start_s = float(row["startTime_ns"]) / 1_000_000_000.0 - base_s
            end_s = float(row["endTime_ns"]) / 1_000_000_000.0 - base_s
            if start_s < 0:
                continue
            if start_s > max_duration_s:
                break
            rows.append({"start_s": round(start_s, 6), "end_s": round(min(end_s, max_duration_s), 6), "label": "speech", "text": row.get("written") or ""})
    return rows


def read_hot3d_objects(path: Path, base_s: float, max_duration_s: float) -> list[dict]:
    rows = []
    seen = set()
    with zipfile.ZipFile(path) as archive:
        mapping = read_hot3d_timecode_mapping(archive)
        handle = archive.open("box2d_objects.csv")
        text = (line.decode("utf-8", errors="replace") for line in handle)
        for row in csv.DictReader(text):
            timecode_ns = int(row["timestamp[ns]"])
            devicetime_ns = mapping.get(timecode_ns)
            if devicetime_ns is None:
                continue
            start_s = devicetime_ns / 1_000_000_000.0 - base_s
            if start_s < 0:
                continue
            if start_s > max_duration_s:
                break
            bucket = round(start_s)
            if bucket in seen:
                continue
            seen.add(bucket)
            rows.append({"start_s": round(start_s, 6), "end_s": round(min(start_s + 1.0, max_duration_s), 6), "label": "object_visible", "text": f"object_uid={row.get('object_uid')}"})
    return rows or [{"start_s": 0.0, "end_s": max_duration_s, "label": "hot3d", "text": "object annotations available"}]


def read_hot3d_timecode_mapping(archive: zipfile.ZipFile) -> dict[int, int]:
    with archive.open("timecode_devicetime_mapping.csv") as handle:
        text = (line.decode("utf-8", errors="replace") for line in handle)
        return {int(row["timecode_ns"]): int(row["devicetime_ns"]) for row in csv.DictReader(text)}


def read_egtea_gaze(path: Path, member: str, max_duration_s: float) -> list[dict]:
    with zipfile.ZipFile(path) as archive:
        lines = archive.open(member).read().decode("utf-8", errors="replace").splitlines()
    header = None
    data = []
    for index, line in enumerate(lines):
        if line.startswith("Time\tType\tTrial"):
            header = line.split("\t")
            data = lines[index + 1 :]
            break
    if not header:
        return []
    rows = []
    base = None
    for line in data:
        parts = line.split("\t")
        if len(parts) < len(header):
            continue
        row = dict(zip(header, parts))
        if row.get("Type") != "SMP":
            continue
        try:
            timestamp = float(row["Time"])
            x_norm = float(row["B POR X [px]"]) / 1280.0
            y_norm = float(row["B POR Y [px]"]) / 960.0
        except ValueError:
            continue
        if base is None:
            base = timestamp
        time_s = (timestamp - base) / 1_000_000.0
        if time_s > max_duration_s:
            break
        rows.append({"time_s": round(time_s, 6), "x_norm": round(clamp(x_norm), 6), "y_norm": round(clamp(y_norm), 6)})
    return rows


def read_egtea_labels(path: Path) -> dict[str, list[dict]]:
    labels: dict[str, list[dict]] = {}
    with zipfile.ZipFile(path) as archive, archive.open("raw_annotations/action_labels.csv") as handle:
        for raw_line in handle:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or line.startswith("#"):
                continue
            parts = [part.strip() for part in line.split(";")]
            if len(parts) < 6:
                continue
            labels.setdefault(parts[2], []).append({"start_ms": float(parts[3]), "end_ms": float(parts[4]), "label": parts[5]})
    for rows in labels.values():
        rows.sort(key=lambda row: row["start_ms"])
    return labels


def max_duration(gaze_rows: list[dict], annotations: list[dict], cap_s: float) -> float:
    values = [float(row["time_s"]) for row in gaze_rows if row.get("time_s") is not None]
    values.extend(float(row["end_s"]) for row in annotations if row.get("end_s") is not None)
    return round(min(max(values or [cap_s]), cap_s), 6)


def write_episode(output_root: Path, dataset: str, episode: str, duration_s: float, files: dict[str, Path]) -> None:
    root = output_root / dataset / episode
    root.mkdir(parents=True, exist_ok=True)
    document = {
        "dataset": dataset,
        "episode_id": episode,
        "duration_s": duration_s,
        "files": {key: str(value if value.is_absolute() else value.name) for key, value in files.items()},
    }
    (root / "episode.json").write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def aria_yaw_pitch_to_normalized(yaw_rads_cpf: float, pitch_rads_cpf: float, fov_degrees: float = 90.0) -> tuple[float, float]:
    # Aria MPS eye-gaze CSVs store angular gaze in CPF. On the undistorted
    # preview videos, positive CPF yaw projects toward image-left.
    # The preview videos are undistorted RGB previews, so this gives a stable
    # normalized 2D approximation without requiring the full Aria calibration stack.
    half_tan = math.tan(math.radians(fov_degrees) / 2.0)
    x = 0.5 - math.tan(yaw_rads_cpf) / (2.0 * half_tan)
    y = 0.5 - math.tan(pitch_rads_cpf) / (2.0 * half_tan)
    return round(clamp(x), 6), round(clamp(y), 6)


if __name__ == "__main__":
    raise SystemExit(main())
