"""Train/val split POINTERS over a clip manifest (gaze-8yc.7 staging).

A "split" here is NOT a copy of the manifest -- it is a set of lightweight POINTERS
(one tiny record per clip: id + dataset + video path) into the joint manifest, so a
downstream trainer joins back to the full rows by ``id``. We never copy clip data and
read the manifest exactly once, streaming only the fields a pointer needs.

Sampling is CLIP-LEVEL and PER-DATASET STRATIFIED: the ratio is applied independently
within each dataset, then the per-dataset buckets are unioned into the aggregate split.
So an 80/20 split takes 80/20 of EACH dataset -- the val split is guaranteed to contain
clips from every (incl. minority) dataset. Deterministic given a seed.

Outputs (under ``<out_dir>/<name>/``):
  <split>.jsonl        one pointer per line: {"id", "dataset", "video"} for that split
  split_index.json     manifest of the split: ratios, seed, per-dataset+total counts,
                       source manifest path + sha, and the relative pointer-file paths.
"""
from __future__ import annotations

import hashlib
import json
import random
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator


def join_manifests(
    sources: list[tuple[str | Path, set[str] | None]],
    out_path: str | Path,
) -> dict[str, Any]:
    """Concatenate per-run manifests into ONE joint manifest.jsonl (streaming).

    ``sources`` is ``[(manifest_path, keep_datasets_or_None), ...]``: from each source
    manifest, keep only rows whose ``dataset`` is in ``keep_datasets`` (None = keep all).
    This drops the stray sample-default contamination (e.g. take only nymeria from the
    nym run, only the 4 real datasets from the nonnym run). Error rows and rows without
    an id are skipped; duplicate ids (same clip appearing twice) are de-duped, first wins.
    Streams line by line -> constant memory. Returns counts by dataset + total.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    by_ds: dict[str, int] = defaultdict(int)
    dropped = 0
    with out_path.open("w", encoding="utf-8") as out_fh:
        for src, keep in sources:
            for line in Path(src).open(encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "error" in row or not row.get("id"):
                    continue
                ds = str(row.get("dataset") or _dataset_from_id(row["id"]))
                if keep is not None and ds not in keep:
                    dropped += 1
                    continue
                if row["id"] in seen:
                    continue
                seen.add(row["id"])
                by_ds[ds] += 1
                out_fh.write(json.dumps(row, sort_keys=True) + "\n")
    return {"out": str(out_path), "total": len(seen),
            "by_dataset": dict(sorted(by_ds.items())), "dropped_off_dataset": dropped}


def iter_clip_pointers(manifest_path: str | Path) -> Iterator[dict[str, str]]:
    """Stream lightweight pointers from a clip manifest.jsonl, one per usable clip.

    Reads line by line and parses each row, but keeps ONLY the pointer fields
    (``id``, ``dataset``, ``video``) -- the heavy per-frame ``points``/``message_list``
    are dropped immediately, so peak memory stays tiny even for a 400k-clip manifest.
    Error rows (``{"error": ...}``) and rows missing an id are skipped.
    """
    p = Path(manifest_path)
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in row or not row.get("id"):
                continue
            yield {
                "id": str(row["id"]),
                "dataset": str(row.get("dataset") or _dataset_from_id(row["id"])),
                "video": str(row.get("video") or ""),
            }


def _dataset_from_id(clip_id: str) -> str:
    """Fallback: clip id is ``<dataset>:<episode>#seg<k>`` -> dataset before the ':'."""
    return clip_id.split(":", 1)[0] if ":" in clip_id else "unknown"


def make_stratified_splits(
    pointers: list[dict[str, str]],
    ratios: dict[str, float],
    *,
    seed: int = 0,
) -> dict[str, list[dict[str, str]]]:
    """Partition clip pointers into named splits, PER-DATASET STRATIFIED at clip level.

    ``ratios`` maps split name -> fraction (e.g. ``{"train": 0.8, "val": 0.2}``); the
    fractions are normalized to sum to 1. Within each dataset the clips are shuffled
    (seeded) and partitioned by the ratios; the per-dataset partitions are then unioned
    across datasets into each named split. Every split that gets a non-zero ratio is
    guaranteed >=1 clip from every dataset that has enough clips (largest-remainder
    rounding gives the leftover clips to the split with the biggest fractional part, so
    tiny/minority datasets still land in val).

    Deterministic: same pointers + ratios + seed -> identical assignment. Each dataset
    uses an independent seeded shuffle derived from (seed, dataset) so adding/removing
    one dataset doesn't reshuffle the others.
    """
    names = list(ratios)
    total_r = sum(ratios.values())
    if total_r <= 0:
        raise ValueError(f"split ratios must sum to > 0: {ratios}")
    norm = {n: ratios[n] / total_r for n in names}

    by_ds: dict[str, list[dict[str, str]]] = defaultdict(list)
    for ptr in pointers:
        by_ds[ptr["dataset"]].append(ptr)

    out: dict[str, list[dict[str, str]]] = {n: [] for n in names}
    for ds in sorted(by_ds):
        clips = sorted(by_ds[ds], key=lambda p: p["id"])  # stable base order
        rng = random.Random(f"{seed}:{ds}")               # per-dataset independent shuffle
        rng.shuffle(clips)
        for n, chunk in _partition_largest_remainder(clips, norm, names).items():
            out[n].extend(chunk)
    return out


def _partition_largest_remainder(
    clips: list[dict[str, str]], norm: dict[str, float], names: list[str]
) -> dict[str, list[dict[str, str]]]:
    """Split ``clips`` into named chunks of size ~ norm[name]*len, summing to len exactly.

    Uses largest-remainder (Hamilton) rounding so every clip is assigned exactly once and
    the leftover from flooring goes to the splits with the largest fractional parts (so a
    1-clip dataset with an 80/20 ratio puts its single clip in train, a 2-clip one puts 1
    in each, etc.).
    """
    n = len(clips)
    if n == 0:
        return {nm: [] for nm in names}
    raw = {nm: norm[nm] * n for nm in names}
    base = {nm: int(raw[nm]) for nm in names}
    leftover = n - sum(base.values())
    # distribute leftover to largest fractional remainders (ties -> ratio order)
    order = sorted(names, key=lambda nm: (raw[nm] - base[nm], norm[nm]), reverse=True)
    for i in range(leftover):
        base[order[i % len(order)]] += 1
    chunks: dict[str, list[dict[str, str]]] = {}
    pos = 0
    for nm in names:
        chunks[nm] = clips[pos : pos + base[nm]]
        pos += base[nm]
    return chunks


def write_splits(
    splits: dict[str, list[dict[str, str]]],
    out_dir: str | Path,
    *,
    name: str,
    ratios: dict[str, float],
    seed: int,
    manifest_path: str | Path,
    manifest_sha: str | None = None,
) -> dict[str, Any]:
    """Write per-split pointer JSONLs + a split_index.json. Returns the index dict."""
    base = Path(out_dir) / name
    base.mkdir(parents=True, exist_ok=True)

    per_split_counts: dict[str, Any] = {}
    files: dict[str, str] = {}
    for split_name, ptrs in splits.items():
        fname = f"{split_name}.jsonl"
        (base / fname).write_text(
            "".join(json.dumps(p, sort_keys=True) + "\n" for p in ptrs), encoding="utf-8"
        )
        files[split_name] = fname
        by_ds: dict[str, int] = defaultdict(int)
        for p in ptrs:
            by_ds[p["dataset"]] += 1
        per_split_counts[split_name] = {"total": len(ptrs), "by_dataset": dict(sorted(by_ds.items()))}

    index = {
        "name": name,
        "kind": "clip_pointer_split",
        "ratios": ratios,
        "seed": seed,
        "stratified_by": "dataset",
        "sample_level": "clip",
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": manifest_sha,
        "files": files,
        "counts": per_split_counts,
        "pointer_fields": ["id", "dataset", "video"],
        "join_key": "id",
    }
    (base / "split_index.json").write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
    return index


def file_sha256(path: str | Path, _bufsize: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(_bufsize), b""):
            h.update(chunk)
    return h.hexdigest()


def build_split(
    manifest_path: str | Path,
    out_dir: str | Path,
    *,
    name: str,
    ratios: dict[str, float],
    seed: int = 0,
) -> dict[str, Any]:
    """End-to-end: stream pointers from the manifest, stratify-split, write pointer files.

    Cheap: one streaming pass over the manifest keeping only pointer fields. Returns the
    split_index dict (also written to ``<out_dir>/<name>/split_index.json``).
    """
    pointers = list(iter_clip_pointers(manifest_path))
    if not pointers:
        raise ValueError(f"no usable clips in manifest: {manifest_path}")
    splits = make_stratified_splits(pointers, ratios, seed=seed)
    sha = file_sha256(manifest_path)
    return write_splits(
        splits, out_dir, name=name, ratios=ratios, seed=seed,
        manifest_path=manifest_path, manifest_sha=sha,
    )


def upload_splits(
    local_split_dir: str | Path,
    *,
    name: str,
    s3_uri: str = "s3://far-research-internal/colbhben/gaze/splits",
    aws_bin: str = "aws",
    on_remote_host: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Sync the split pointer dir to S3 under ``<s3_uri>/<name>/``.

    Local AWS access is read-only in some envs; pass ``on_remote_host`` to run the sync
    via ssh on a host with write access (it scp's the small pointer dir over first).
    """
    src = Path(local_split_dir) / name
    dest = f"{s3_uri.rstrip('/')}/{name}/"
    plan = {"local": str(src), "s3": dest, "via": on_remote_host or "local"}
    if dry_run:
        plan["steps"] = (
            [f"ssh {on_remote_host} '{aws_bin} s3 sync <staged>/ {dest}'"] if on_remote_host
            else [f"{aws_bin} s3 sync {src}/ {dest}"]
        )
        return plan
    if on_remote_host:
        remote_tmp = f"/tmp/gaze_split_upload_{name}"
        subprocess.run(["ssh", on_remote_host, f"rm -rf {remote_tmp} && mkdir -p {remote_tmp}"], check=True)
        subprocess.run(["scp", "-rq", f"{src}/.", f"{on_remote_host}:{remote_tmp}/"], check=True)
        out = subprocess.run(
            ["ssh", on_remote_host, f"{aws_bin} s3 sync {remote_tmp}/ {dest}"],
            capture_output=True, text=True,
        )
        subprocess.run(["ssh", on_remote_host, f"rm -rf {remote_tmp}"], check=False)
    else:
        out = subprocess.run([aws_bin, "s3", "sync", f"{src}/", dest], capture_output=True, text=True)
    plan["returncode"] = out.returncode
    plan["stderr_tail"] = out.stderr.splitlines()[-5:]
    return plan
