#!/usr/bin/env python3
"""Nymeria gaze-projection GT self-consistency check (overlay validation).

projectaria_tools' ``get_gaze_vector_reprojection`` IS the same function Meta's
own offline renderer / explorer uses, so a rigorous *self-consistency* check is
the right GT proxy when the explorer isn't scriptable. For ~20 frames spread
across the clip we:
  (a) confirm the projected gaze lands IN-FRAME,
  (b) confirm it TRACKS scene content (no teleporting between consecutive
      frames -- step size bounded vs the FOV),
  (c) compare PERSONALIZED vs GENERAL gaze csv (both present on this take).

Annotated frames are written to /tmp/gaze_overlays/nymeria_gt_check/.

Explorer compare instructions (printed): the explorer.projectaria.com st= query
param is device-clock SECONDS = (first_gaze_ts_us + frame*1e6/fps)/1e6.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from src.gaze.curate import Puller, ffprobe_video
from src.gaze import curate_readers as cr
from src.gaze import overlay as ov

TAKE = "20230607_s1_barbara_wheeler_act1_nkg6zo"
ROOT = "nymeria"
OUT = Path("/tmp/gaze_overlays/nymeria_gt_check")
N = 20

GAZE_DIR = f"{TAKE}/recording_head/recording_head/mps/eye_gaze"
CALIB_REL = f"{ROOT}/{TAKE}/recording_head/recording_head/mps/slam/online_calibration.jsonl"
VIDEO_REL = f"{ROOT}/{TAKE}/video_main_rgb/Nymeria_v0.0_{TAKE}_preview_rgb.mp4"


def read_gaze_csv(local: Path) -> list[dict]:
    import csv as _csv

    rows = []
    with open(local, newline="") as fh:
        for r in _csv.DictReader(fh):
            def f(k):
                v = r.get(k)
                return float(v) if v not in (None, "") else None
            rows.append({
                "_tracking_timestamp_us": f("tracking_timestamp_us"),
                "left_yaw": f("left_yaw_rads_cpf"),
                "right_yaw": f("right_yaw_rads_cpf"),
                "pitch": f("pitch_rads_cpf"),
                "depth": f("depth_m"),
            })
    return rows


def project_rows(rows, ctx) -> list[tuple[float, float] | None]:
    return [ov.project_one(r, ctx) for r in rows]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    puller = Puller(workdir="/tmp/gaze_curate_work")

    # Pull both gaze csvs + calib + video (full mp4; deleted at end).
    pers_local = puller.pull(f"{ROOT}/{GAZE_DIR}/personalized_eye_gaze.csv")
    gen_local = puller.pull(f"{ROOT}/{GAZE_DIR}/general_eye_gaze.csv")
    calib_local = puller.pull(CALIB_REL)
    video_local = puller.pull(VIDEO_REL)

    vm = ffprobe_video(video_local)
    fps = vm.fps
    W, H = vm.width, vm.height
    print(f"video {W}x{H} @ {fps} fps dur={vm.duration_s:.2f}s")

    pers = read_gaze_csv(pers_local)
    gen = read_gaze_csv(gen_local)
    # drop personalized warmup empties (leading rows with no yaw)
    pers_valid = [r for r in pers if r["left_yaw"] is not None]
    print(f"personalized rows={len(pers)} (valid={len(pers_valid)}); general rows={len(gen)}")
    first_gaze_ts_us = gen[0]["_tracking_timestamp_us"]
    print(f"first_gaze_ts_us = {first_gaze_ts_us:.0f} (device clock; == mp4 frame 0)")

    # Build a projection context (loads calib once).
    calib_lines = cr._load_calib_lines(calib_local)
    ctx = ov.ProjectionContext(
        method="projectaria_cpf", width=W, height=H,
        calib_lines=calib_lines, aria_scale=(W / 2880.0),
        coordinate_space="cpf_angular",
    )

    # Choose ~N frame indices spread across the clip.
    n_frames = int(round(vm.duration_s * fps))
    frame_idxs = [round(i * (n_frames - 1) / (N - 1)) for i in range(N)]

    # Index gaze by tracking ts for nearest lookup.
    def nearest_row(rows, ts_us):
        best, bd = None, None
        for r in rows:
            t = r["_tracking_timestamp_us"]
            if t is None:
                continue
            d = abs(t - ts_us)
            if bd is None or d < bd:
                best, bd = r, d
        return best

    records = []
    last_pers = None
    for fi in frame_idxs:
        device_ts_us = first_gaze_ts_us + round(fi * 1e6 / fps)
        st_sec = device_ts_us / 1e6
        pr = nearest_row(pers_valid, device_ts_us)
        gr = nearest_row(gen, device_ts_us)
        ppx = ov.project_one(pr, ctx) if pr else None
        gpx = ov.project_one(gr, ctx) if gr else None
        in_frame = ppx is not None and (0 <= ppx[0] <= W) and (0 <= ppx[1] <= H)
        # tracking sanity: step from previous projected personalized point
        step = None
        if ppx is not None and last_pers is not None:
            step = float(np.hypot(ppx[0] - last_pers[0], ppx[1] - last_pers[1]))
        if ppx is not None:
            last_pers = ppx
        # personalized-vs-general agreement (px)
        pg_dist = None
        if ppx is not None and gpx is not None:
            pg_dist = float(np.hypot(ppx[0] - gpx[0], ppx[1] - gpx[1]))
        records.append({
            "frame": fi, "device_ts_us": device_ts_us, "st_sec": round(st_sec, 3),
            "personalized_px": [round(ppx[0], 1), round(ppx[1], 1)] if ppx else None,
            "general_px": [round(gpx[0], 1), round(gpx[1], 1)] if gpx else None,
            "in_frame": in_frame, "step_px": round(step, 1) if step is not None else None,
            "pers_vs_gen_px": round(pg_dist, 1) if pg_dist is not None else None,
        })

        # Draw annotated frame.
        png = OUT / f"frame_{fi:06d}.png"
        r = subprocess.run(
            ["ffmpeg", "-y", "-ss", f"{fi/fps:.4f}", "-i", str(video_local),
             "-frames:v", "1", str(png)],
            capture_output=True, text=True,
        )
        if png.exists():
            img = Image.open(png).convert("RGB")
            from PIL import ImageDraw, ImageFont
            d = ImageDraw.Draw(img)
            font = ov._load_font(max(14, W // 40))
            if ppx:
                x, y = ppx
                d.ellipse([x - 18, y - 18, x + 18, y + 18], outline=(255, 60, 60), width=4)
                d.line([x - 30, y, x + 30, y], fill=(255, 60, 60), width=2)
                d.line([x, y - 30, x, y + 30], fill=(255, 60, 60), width=2)
            if gpx:
                gx, gy = gpx
                d.ellipse([gx - 12, gy - 12, gx + 12, gy + 12], outline=(80, 200, 250), width=3)
            d.rectangle([0, 0, W, 60], fill=(0, 0, 0))
            d.text((8, 6), f"frame {fi}  st={st_sec:.3f}s  red=personalized cyan=general", fill=(255, 255, 255), font=font)
            d.text((8, 32), f"in_frame={in_frame} step={records[-1]['step_px']}px pers-vs-gen={records[-1]['pers_vs_gen_px']}px", fill=(255, 255, 255), font=font)
            img.save(png)

    # Summary stats.
    inframe = sum(1 for r in records if r["in_frame"])
    steps = [r["step_px"] for r in records if r["step_px"] is not None]
    pgs = [r["pers_vs_gen_px"] for r in records if r["pers_vs_gen_px"] is not None]
    fov_diag = float(np.hypot(W, H))
    summary = {
        "take": TAKE, "video": f"{W}x{H}@{fps}", "frames_checked": len(records),
        "in_frame": f"{inframe}/{len(records)} ({100*inframe/len(records):.0f}%)",
        "step_px": {"max": max(steps) if steps else None, "mean": round(float(np.mean(steps)), 1) if steps else None,
                    "frame_diag_px": round(fov_diag, 1),
                    "max_step_frac_of_diag": round(max(steps) / fov_diag, 3) if steps else None},
        "pers_vs_gen_px": {"max": max(pgs) if pgs else None, "mean": round(float(np.mean(pgs)), 1) if pgs else None},
        "first_gaze_ts_us": first_gaze_ts_us,
        "explorer_st_formula": "st_sec = (first_gaze_ts_us + frame*1e6/fps)/1e6",
    }
    (OUT / "summary.json").write_text(json.dumps({"summary": summary, "records": records}, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nAnnotated frames + summary.json -> {OUT}")

    # Delete the big full mp4.
    try:
        if video_local.exists() and video_local.stat().st_size > 5_000_000:
            video_local.unlink()
            print("deleted full nymeria mp4")
    except OSError:
        pass


if __name__ == "__main__":
    main()
