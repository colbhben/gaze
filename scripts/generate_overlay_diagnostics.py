#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess

from gaze.table import read_table
from gaze.video import find_ffmpeg


def main() -> int:
    parser = argparse.ArgumentParser(description="Render sampled video frames with rectified gaze points burned in.")
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sample", action="append", default=[], help="Episode and times, e.g. aea/ep:10,20,30")
    args = parser.parse_args()

    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise SystemExit("ffmpeg not found")
    args.output_root.mkdir(parents=True, exist_ok=True)
    for sample in args.sample:
        episode, _, raw_times = sample.partition(":")
        times = [float(item) for item in raw_times.split(",") if item]
        render_episode(args.canonical_root, args.output_root, ffmpeg, episode, times)
    return 0


def render_episode(canonical_root: Path, output_root: Path, ffmpeg: str, episode: str, times: list[float]) -> None:
    episode_root = canonical_root / "episodes" / episode
    doc = json.loads((episode_root / "episode.json").read_text(encoding="utf-8"))
    video_path = episode_root / doc["files"]["video"]
    gaze = [row for row in read_table(episode_root / doc["files"]["gaze"]) if row.get("x_norm") is not None and row.get("y_norm") is not None]
    for time_s in times:
        row = min(gaze, key=lambda item: abs(float(item["time_s"]) - time_s))
        x = max(0, min(243, round(float(row["x_norm"]) * 244)))
        y = max(0, min(223, round(float(row["y_norm"]) * 224)))
        output = output_root / f"{episode.replace('/', '_')}_{time_s:06.1f}.png"
        draw = f"drawbox=x={max(x - 8, 0)}:y={max(y - 8, 0)}:w=16:h=16:color=red@0.85:t=3"
        subprocess.run(
            [ffmpeg, "-y", "-ss", str(time_s), "-i", str(video_path), "-frames:v", "1", "-vf", draw, str(output)],
            check=True,
            capture_output=True,
            text=True,
        )
        print(output)
if __name__ == "__main__":
    raise SystemExit(main())
