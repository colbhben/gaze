#!/usr/bin/env python3
"""Render gaze+annotation overlay clips for all 7 sample episodes.

Usage:
    .venv/bin/python scripts/render_overlays.py [slug ...]

Writes /tmp/gaze_overlays/<slug>.mp4 (<=20s, <=720p). Big source mp4s are
pulled, trimmed, then deleted (only small egome/egtea full mp4s are kept by
the puller cache). For nymeria/hd-epic a window with annotation activity is
chosen automatically (override with START below).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gaze.curate import Puller
from src.gaze import overlay as ov

OUT = Path("/tmp/gaze_overlays")
SAMPLES = json.loads(
    (Path(__file__).resolve().parents[1] / "recipes" / "_sample_episodes.json").read_text()
)["samples"]

# Optional explicit window starts (else auto-picked around first annotation).
START = {"nymeria": 95.0}


def main(only: list[str] | None = None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    puller = Puller(workdir="/tmp/gaze_curate_work")
    table = []
    for slug, info in SAMPLES.items():
        if only and slug not in only:
            continue
        ep = info["episode_id"]
        extra = {k: v for k, v in info.items() if k in ("session", "participant", "take_name")}
        t0 = time.time()
        res = ov.render_overlay(
            slug, ep, puller, OUT / f"{slug}.mp4",
            max_seconds=20.0, start_s=START.get(slug), sample_extra=extra,
        )
        dt = time.time() - t0
        row = {
            "slug": slug, "dur_s": round(res.duration_s, 2),
            "dims": f"{res.width}x{res.height}", "frames": res.frames,
            "gaze_visible": f"{res.gaze_visible_frames}/{res.frames}",
            "chans": res.channels_shown, "render_s": round(dt, 1),
        }
        table.append(row)
        print(json.dumps(row), flush=True)
    print("\n=== SUMMARY ===")
    for r in table:
        print(f"  {r['slug']:12s} {r['dur_s']:6.2f}s {r['dims']:>10s} "
              f"gaze {r['gaze_visible']:>9s} chans={r['chans']}")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
