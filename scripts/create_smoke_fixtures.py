#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DATASETS = ["holoassist", "ego-exo4d", "hot3d", "nymeria", "aea", "egtea"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Create tiny distinct raw episodes for pipeline smoke tests.")
    parser.add_argument("--root", default="/nfs/colbhben/gaze/unprocessed")
    parser.add_argument("--datasets", default=",".join(DATASETS))
    args = parser.parse_args()
    root = Path(args.root)
    for index, dataset in enumerate(item.strip() for item in args.datasets.split(",") if item.strip()):
        create_fixture(root, dataset, index)
    return 0


def create_fixture(base: Path, dataset: str, index: int) -> None:
    episode_id = f"smoke_{dataset.replace('-', '_')}"
    root = base / dataset / episode_id
    root.mkdir(parents=True, exist_ok=True)
    (root / "video.mp4").write_bytes(f"synthetic non-video placeholder for {dataset}\n".encode("utf-8"))

    gaze_offset = index * 0.07
    with (root / "gaze.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["time_s", "x_norm", "y_norm", "x_px", "y_px"])
        writer.writeheader()
        for frame in range(5):
            x_norm = min(0.95, 0.08 + gaze_offset + frame * 0.08)
            y_norm = min(0.95, 0.18 + (index % 3) * 0.16 + frame * 0.05)
            writer.writerow(
                {
                    "time_s": frame / 10,
                    "x_norm": round(x_norm, 4),
                    "y_norm": round(y_norm, 4),
                    "x_px": round(x_norm * 1280, 2),
                    "y_px": round(y_norm * 720, 2),
                }
            )

    with (root / "annotations.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["start_s", "end_s", "label", "text"])
        writer.writeheader()
        writer.writerow({"start_s": 0.0, "end_s": 0.2, "label": f"{dataset}-setup", "text": f"{dataset} smoke start"})
        writer.writerow({"start_s": 0.2, "end_s": 0.4, "label": f"{dataset}-finish", "text": f"{dataset} smoke finish"})

    with (root / "depth.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["time_s", "depth_m"])
        writer.writeheader()
        for frame in range(3):
            writer.writerow({"time_s": frame / 5, "depth_m": round(1.0 + index * 0.1 + frame * 0.2, 4)})

    (root / "episode.json").write_text(
        json.dumps(
            {
                "dataset": dataset,
                "episode_id": episode_id,
                "duration_s": 0.4,
                "files": {"video": "video.mp4", "gaze": "gaze.csv", "annotations": "annotations.csv", "depth": "depth.csv"},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(root)


if __name__ == "__main__":
    raise SystemExit(main())
