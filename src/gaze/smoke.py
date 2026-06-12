"""Assemble a viewer-ready smoke manifest from extracted sample episodes.

Builds a canonical-style root the existing `gaze serve` viewer understands:

    <root>/
      manifest.jsonl                       # episode index (dataset, episode_id, ...)
      manifest.parquet[.jsonl]             # same, table form the server reads
      episodes/<dataset>/<episode_id>/
        episode.json                       # {dataset, episode_id, files{video,gaze,annotations,...}, gaze_meta, ...}
        overlay.mp4                        # gaze+annotation overlay clip (the "video")
        gaze.jsonl                         # reconciled gaze rows (video-zero seconds)
        annotations.jsonl                  # flattened raw segments across channels
        bundle.json                        # full EpisodeBundle.to_dict() for provenance

The smoke manifest is uploaded to S3 (smoke_manifest/ prefix) via the remote
host, since local AWS access is read-denied here. Side-by-side GT clips are
included where available.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .table import write_table
from .curate import load_recipe


def _reconcile(raw_t: float | None, transform: str, *, gaze_t0: float | None, clip_start_s: float | None) -> float | None:
    """Rebase a raw channel time onto the video-zero clock per epoch_sync.

    Mirrors overlay.reconcile_to_video_clock so the viewer manifest and the
    burned-in overlay share one clock.
    """
    if raw_t is None:
        return None
    if transform == "as_is":
        return raw_t
    if transform == "subtract_first_gaze":
        return raw_t - (gaze_t0 or 0.0)
    if transform == "subtract_clip_start":
        return raw_t - (clip_start_s or 0.0)
    return raw_t


def _clip_start_s(slug: str, episode_id: str) -> float | None:
    """egtea: clip start seconds = filename start_ms / 1000."""
    if slug != "egtea":
        return None
    m = re.search(r"-(\d+)-(\d+)-F\d+-F\d+$", episode_id)
    return int(m.group(1)) / 1000.0 if m else None


def build_smoke_manifest(
    extract_dir: str | Path,
    overlays_dir: str | Path,
    out_root: str | Path,
    *,
    datasets: list[str] | None = None,
) -> dict[str, Any]:
    """Assemble the viewer-ready root from /tmp/gaze_extract + /tmp/gaze_overlays."""
    extract_dir = Path(extract_dir)
    overlays_dir = Path(overlays_dir)
    out_root = Path(out_root)
    (out_root / "episodes").mkdir(parents=True, exist_ok=True)

    bundles = sorted(extract_dir.glob("*.json"))
    bundles = [b for b in bundles if not b.stem.endswith("_full")]
    if datasets:
        bundles = [b for b in bundles if b.stem in set(datasets)]

    index: list[dict[str, Any]] = []
    for bundle_path in bundles:
        slug = bundle_path.stem
        summary = json.loads(bundle_path.read_text(encoding="utf-8"))
        full_path = extract_dir / f"{slug}_full.json"
        full = json.loads(full_path.read_text(encoding="utf-8")) if full_path.exists() else {}
        episode_id = summary.get("episode_id", slug)
        safe_ep = _safe(episode_id)
        ep_dir = out_root / "episodes" / slug / safe_ep
        ep_dir.mkdir(parents=True, exist_ok=True)

        files: dict[str, str] = {}

        # epoch_sync: rebase gaze + annotation times onto the video-zero clock
        # (same reconciliation the overlay uses), so the viewer table matches the video.
        try:
            recipe = load_recipe(slug)
            epoch = recipe.get("epoch_sync") or {}
        except Exception:
            epoch = {}
        gaze_transform = (epoch.get("gaze") or {}).get("transform", "as_is")
        anno_transform = (epoch.get("annotations") or {}).get("transform", "as_is")

        gaze = full.get("gaze") or {}
        gaze_rows = gaze.get("rows") or []
        gaze_t0 = next((r["t_s"] for r in gaze_rows if r.get("t_s") is not None), 0.0)
        clip_start = _clip_start_s(slug, episode_id)

        # overlay video
        overlay_src = overlays_dir / f"{slug}.mp4"
        if overlay_src.exists():
            shutil.copy2(overlay_src, ep_dir / "overlay.mp4")
            files["video"] = "overlay.mp4"
        sbs_src = overlays_dir / f"{slug}_side_by_side.mp4"
        if sbs_src.exists():
            shutil.copy2(sbs_src, ep_dir / "side_by_side.mp4")
            files["side_by_side"] = "side_by_side.mp4"

        # gaze rows (times reconciled to video-zero seconds)
        if gaze_rows:
            thinned = _thin_gaze(gaze_rows)
            for r in thinned:
                if "t_s" in r:
                    r["t_s"] = _reconcile(r.get("t_s"), gaze_transform, gaze_t0=gaze_t0, clip_start_s=clip_start)
            write_table(thinned, ep_dir / "gaze.jsonl")
            files["gaze"] = "gaze.jsonl"

        # annotations flattened across channels (times reconciled to video-zero seconds)
        anno_rows = _flatten_annotations(full.get("annotations") or summary.get("annotations") or [])
        for r in anno_rows:
            for key in ("start_s", "end_s", "point_s"):
                r[key] = _reconcile(r.get(key), anno_transform, gaze_t0=gaze_t0, clip_start_s=clip_start)
        # re-sort after reconciliation (offsets are monotonic so order is preserved, but be safe)
        anno_rows.sort(key=lambda r: (r.get("start_s") if r.get("start_s") is not None else (r.get("point_s") or 0.0)))
        if anno_rows:
            write_table(anno_rows, ep_dir / "annotations.jsonl")
            files["annotations"] = "annotations.jsonl"

        # provenance bundle
        (ep_dir / "bundle.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

        episode_doc = {
            "dataset": slug,
            "episode_id": episode_id,
            "files": files,
            "video_meta": summary.get("video"),
            "gaze_meta": {k: v for k, v in (gaze or {}).items() if k != "rows"},
            "annotation_channels": [a.get("name") for a in (summary.get("annotations") or [])],
            "warnings": summary.get("warnings", []),
        }
        (ep_dir / "episode.json").write_text(json.dumps(episode_doc, indent=2, sort_keys=True), encoding="utf-8")

        index.append({
            "id": f"{slug}:{episode_id}",
            "dataset": slug,
            "episode_id": episode_id,
            "modalities": ",".join(
                ["video"] * ("video" in files) + ["gaze"] * ("gaze" in files)
                + (["annotations"] if "annotations" in files else [])
            ),
            "duration_s": (summary.get("video") or {}).get("duration_s"),
            "gaze_space": (gaze or {}).get("coordinate_space"),
            "annotation_channels": ";".join(a.get("name") for a in (summary.get("annotations") or [])),
            "has_side_by_side": "side_by_side" in files,
        })

    # manifest in both jsonl and the table form the server reads
    (out_root / "manifest.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in index) + "\n", encoding="utf-8"
    )
    write_table(index, out_root / "manifest.parquet")

    report = {
        "root": str(out_root),
        "episodes": len(index),
        "datasets": [row["dataset"] for row in index],
        "with_side_by_side": [row["dataset"] for row in index if row["has_side_by_side"]],
    }
    (out_root / "smoke_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def upload_smoke_manifest(
    local_root: str | Path,
    *,
    ssh_host: str = "sumedhso-L40S",
    s3_uri: str = "s3://far-research-internal/colbhben/gaze/unprocessed/smoke_manifest",
    remote_tmp: str = "/tmp/gaze_smoke_upload",
    aws_bin: str = "/snap/bin/aws",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Push the local smoke root to S3 via the remote host (local AWS is read-only).

    scp the tree to the remote, then `aws s3 sync` it to the bucket prefix.
    """
    local_root = Path(local_root)
    plan = {
        "local_root": str(local_root),
        "ssh_host": ssh_host,
        "s3_uri": s3_uri,
        "remote_tmp": remote_tmp,
        "dry_run": dry_run,
    }
    if dry_run:
        plan["steps"] = [
            f"ssh {ssh_host} 'rm -rf {remote_tmp} && mkdir -p {remote_tmp}'",
            f"scp -rq {local_root}/. {ssh_host}:{remote_tmp}/",
            f"ssh {ssh_host} '{aws_bin} s3 sync {remote_tmp}/ {s3_uri}/'",
        ]
        return plan
    subprocess.run(["ssh", ssh_host, f"rm -rf {remote_tmp} && mkdir -p {remote_tmp}"], check=True)
    subprocess.run(["scp", "-rq", f"{local_root}/.", f"{ssh_host}:{remote_tmp}/"], check=True)
    out = subprocess.run(
        ["ssh", ssh_host, f"{aws_bin} s3 sync {remote_tmp}/ {s3_uri}/"],
        capture_output=True, text=True,
    )
    plan["returncode"] = out.returncode
    plan["uploaded"] = [line for line in out.stdout.splitlines() if "upload:" in line][-20:]
    plan["stderr_tail"] = out.stderr.splitlines()[-5:]
    # cleanup remote staging
    subprocess.run(["ssh", ssh_host, f"rm -rf {remote_tmp}"], check=False)
    return plan


def _safe(name: str) -> str:
    return name.replace("/", "__").replace(":", "_")


def _thin_gaze(rows: list[dict[str, Any]], max_rows: int = 20000) -> list[dict[str, Any]]:
    """Keep the core gaze columns; subsample if very long (viewer-friendly)."""
    keep = ("t_s", "x", "y", "x_norm", "y_norm", "x_px", "y_px", "valid", "depth", "yaw", "pitch")
    thinned = [{k: r[k] for k in keep if k in r} for r in rows]
    if len(thinned) > max_rows:
        step = len(thinned) // max_rows + 1
        thinned = thinned[::step]
    return thinned


def _flatten_annotations(channels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for ch in channels:
        name = ch.get("name")
        segs = ch.get("segments")
        if not isinstance(segs, list):
            continue
        for s in segs:
            out.append({
                "channel": name,
                "kind": ch.get("kind"),
                "start_s": s.get("start_s"),
                "end_s": s.get("end_s"),
                "point_s": s.get("point_s"),
                "text": s.get("text"),
            })
    out.sort(key=lambda r: (r.get("start_s") if r.get("start_s") is not None else (r.get("point_s") or 0.0)))
    return out


# --------------------------------------------------------------------------- #
# Molmo2 manifest -> gaze-serve viewer layout (one viewer-episode per segment).
# --------------------------------------------------------------------------- #
def molmo2_to_viewer_layout(manifest_root: str | Path, out_root: str | Path) -> dict[str, Any]:
    """Convert a molmo2 manifest root into the canonical layout `gaze serve` reads.

    Each manifest row (a clip SEGMENT) becomes one viewer episode under
    ``episodes/<dataset>/<episode_id>__seg<k>/`` with the segment mp4 as ``video``,
    a per-frame gaze table (clip-relative seconds + pixel x/y), and the annotation
    text. Writes ``manifest.parquet`` + ``manifest.jsonl`` the server lists.
    """
    manifest_root = Path(manifest_root)
    out_root = Path(out_root)
    (out_root / "episodes").mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in (manifest_root / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    index: list[dict[str, Any]] = []
    for r in rows:
        if "error" in r or not r.get("video"):
            continue
        ds = r["dataset"]
        ep_seg = f"{_safe(r['episode_id'])}__seg{r.get('seg_index', 0)}"
        ep_dir = out_root / "episodes" / ds / ep_seg
        ep_dir.mkdir(parents=True, exist_ok=True)
        files: dict[str, str] = {}

        src_video = manifest_root / r["video"]
        if src_video.exists():
            shutil.copy2(src_video, ep_dir / "video.mp4")
            files["video"] = "video.mp4"

        # per-frame gaze table (clip-relative s, pixel x/y on the resolution frame)
        gaze_rows = []
        for t, pts in zip(r.get("timestamps", []), r.get("points", [])):
            if pts:
                gaze_rows.append({"time_s": t, "x_px": pts[0]["x"], "y_px": pts[0]["y"]})
            else:
                gaze_rows.append({"time_s": t, "x_px": None, "y_px": None})
        if gaze_rows:
            write_table(gaze_rows, ep_dir / "gaze.jsonl")
            files["gaze"] = "gaze.jsonl"

        md = r.get("metadata") or {}
        anno_text = md.get("annotation_text")
        anno_channel = md.get("annotation_channel")
        seg_end = md.get("clip_end_time")
        seg_start = md.get("clip_start_time")
        dur = (seg_end - seg_start) if (seg_end is not None and seg_start is not None) else None
        anno_rows: list[dict[str, Any]] = []
        # FINAL training annotation (assembled from source + aux per the recipe
        # annotation_bundle policy). Shown first; spans the whole clip.
        final_anno = md.get("final_annotation")
        if final_anno and final_anno != anno_text:
            anno_rows.append({
                "time_s": 0.0,
                "end_s": dur,
                "role": "final",
                "channel": "(bundle)",
                "label": "(bundle)",
                "text": final_anno,
            })
        # SOURCE channel: the annotation that drove the clip (spans the whole clip).
        if anno_text:
            anno_rows.append({
                "time_s": 0.0,
                "end_s": dur,
                "role": "source",
                "channel": anno_channel,
                "label": anno_channel,   # the channel that drove this clip
                "text": anno_text,
            })
        # AUXILIARY channels: every other channel temporally covering the clip, at its
        # own clip-relative [start_s, end_s] (item 4). Carries channel + role so the
        # viewer can mark which annotation is the source vs auxiliary.
        for aux in (md.get("auxiliary_annotations") or []):
            anno_rows.append({
                "time_s": aux.get("start_s", 0.0),
                "end_s": aux.get("end_s", dur),
                "role": "auxiliary",
                "channel": aux.get("channel"),
                "label": aux.get("channel"),
                "text": aux.get("text"),
                "source_duration_s": aux.get("source_duration_s"),
                "overlap_s": aux.get("overlap_s"),
            })
        if anno_rows:
            write_table(anno_rows, ep_dir / "annotations.jsonl")
            files["annotations"] = "annotations.jsonl"

        (ep_dir / "episode.json").write_text(json.dumps({
            "dataset": ds,
            "episode_id": ep_seg,
            "files": files,
            "resolution": r.get("resolution"),
            "fps": r.get("fps"),
            "num_frames_real": r.get("num_frames_real"),
            "metadata": r.get("metadata"),
        }, indent=2, sort_keys=True), encoding="utf-8")

        index.append({
            "id": f"{ds}:{ep_seg}",
            "dataset": ds,
            "episode_id": ep_seg,
            "modalities": ",".join(k for k in ("video", "gaze", "annotations") if k in files),
            "clip_start_time": (r.get("metadata") or {}).get("clip_start_time"),
            "clip_end_time": (r.get("metadata") or {}).get("clip_end_time"),
            "num_frames_real": r.get("num_frames_real"),
        })

    (out_root / "manifest.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in index) + "\n", encoding="utf-8"
    )
    write_table(index, out_root / "manifest.parquet")
    report = {"root": str(out_root), "episodes": len(index),
              "datasets": sorted({row["dataset"] for row in index})}
    (out_root / "smoke_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report
