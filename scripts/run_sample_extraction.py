#!/usr/bin/env python3
"""Run extract_episode on all 7 sample episodes -> /tmp/gaze_extract/.

For each dataset writes:
  <slug>.json        -- EpisodeBundle.to_dict() summary
  <slug>_full.json   -- full gaze rows + annotation segments + projection samples

For projection datasets (nymeria/hd-epic/holoassist + the normalize/already datasets)
projects ~10 evenly-spaced gaze samples and records the pixel coordinates.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gaze.curate import Puller
from src.gaze import curate_readers as cr

OUT = Path("/tmp/gaze_extract")
OUT.mkdir(parents=True, exist_ok=True)

SAMPLES = json.loads((Path(__file__).resolve().parents[1] / "recipes" / "_sample_episodes.json").read_text())["samples"]


def main(only: list[str] | None = None):
    puller = Puller(workdir="/tmp/gaze_curate_work")
    results = {}
    for slug, info in SAMPLES.items():
        if only and slug not in only:
            continue
        ep = info["episode_id"]
        extra = {k: v for k, v in info.items() if k in ("session", "participant", "take_name")}
        print(f"\n===== {slug} :: {ep} =====", flush=True)
        t0 = time.time()
        bundle = cr.extract_episode(slug, ep, puller, sample_extra=extra)
        dt = time.time() - t0
        summary = bundle.to_dict()
        # projection
        proj = None
        if bundle.gaze is not None and bundle.gaze.sample_count > 0:
            try:
                proj = cr.project_gaze(bundle.gaze, bundle.video, puller=puller,
                                       root=cr.load_recipe(slug)["root"],
                                       tok=cr._episode_tokens(slug, ep, extra), n_samples=10)
            except Exception as e:
                import traceback
                proj = {"method": "error", "error": str(e)}
                traceback.print_exc()
        summary["projection_samples"] = proj
        (OUT / f"{slug}.json").write_text(json.dumps(summary, indent=2, default=str))

        # full table dump
        full = {
            "dataset": slug,
            "episode_id": ep,
            "video": asdict(bundle.video) if bundle.video else None,
            "gaze": {
                "coordinate_space": bundle.gaze.coordinate_space if bundle.gaze else None,
                "columns": bundle.gaze.columns if bundle.gaze else None,
                "hz": bundle.gaze.hz if bundle.gaze else None,
                "sample_count": bundle.gaze.sample_count if bundle.gaze else 0,
                "valid_fraction": bundle.gaze.valid_fraction if bundle.gaze else None,
                "rows": bundle.gaze.rows if bundle.gaze else [],
                "notes": bundle.gaze.notes if bundle.gaze else [],
                "source_path": bundle.gaze.source_path if bundle.gaze else None,
            } if bundle.gaze else None,
            "annotations": [
                {
                    "name": a.name, "kind": a.kind, "segment_count": a.segment_count,
                    "coverage_s": a.coverage_s, "coverage_fraction": a.coverage_fraction,
                    "mean_rate_hz": a.mean_rate_hz, "first_start_s": a.first_start_s,
                    "last_end_s": a.last_end_s, "source_path": a.source_path,
                    "segments": a.segments,
                }
                for a in bundle.annotations
            ],
            "projection": proj,
            "warnings": bundle.warnings,
            "emitted": bundle.emitted,
            "emit_reason": bundle.emit_reason,
        }
        (OUT / f"{slug}_full.json").write_text(json.dumps(full, indent=2, default=str))

        results[slug] = summary
        v = bundle.video
        g = bundle.gaze
        print(f"  took {dt:.1f}s | emitted={bundle.emitted} reason={bundle.emit_reason}", flush=True)
        if v:
            print(f"  video: {v.width}x{v.height} @ {v.fps} fps, dur={v.duration_s}s codec={v.codec}", flush=True)
        if g:
            print(f"  gaze: {g.coordinate_space} n={g.sample_count} hz={g.hz} valid={g.valid_fraction}", flush=True)
        for a in bundle.annotations:
            print(f"  anno[{a.name}]: {a.segment_count} segs cov={a.coverage_s} first={a.first_start_s} last={a.last_end_s}", flush=True)
        if proj and proj.get("samples"):
            inframe = sum(1 for s in proj["samples"] if s.get("in_frame"))
            print(f"  proj[{proj['method']}]: {inframe}/{len(proj['samples'])} in-frame; e.g. {proj['samples'][len(proj['samples'])//2]}", flush=True)
        if bundle.warnings:
            for w in bundle.warnings:
                print(f"  WARN: {w}", flush=True)
    return results


if __name__ == "__main__":
    only = sys.argv[1:] or None
    main(only)
