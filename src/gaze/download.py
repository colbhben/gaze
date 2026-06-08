from __future__ import annotations

import concurrent.futures
import os
import re
import subprocess
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
import threading
import time
from typing import Callable
import urllib.error
import urllib.request

from .datasets import Asset, DatasetCatalog, filter_assets, summarize_assets


ProgressCallback = Callable[[dict], None]


@dataclass
class LinkCheck:
    dataset: str
    sequence_id: str
    asset_key: str
    url: str
    ok: bool
    status: int | None = None
    content_type: str | None = None
    error: str | None = None


def plan_downloads(
    catalog: DatasetCatalog,
    datasets: set[str] | None = None,
    modalities: set[str] | None = None,
    sequences: set[str] | None = None,
) -> list[Asset]:
    return filter_assets(catalog.manifest_assets(), datasets=datasets, modalities=modalities, sequences=sequences)


def write_download_manifest(assets: list[Asset], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps([asdict(asset) for asset in assets], indent=2, sort_keys=True), encoding="utf-8")
    return output


def estimate_downloads(assets: list[Asset]) -> list[dict]:
    return summarize_assets(assets)


def verify_links(assets: list[Asset], sample_per_dataset: int = 3, timeout_s: float = 15.0) -> list[LinkCheck]:
    selected: list[Asset] = []
    counts: dict[str, int] = {}
    for asset in assets:
        if not asset.url:
            continue
        if sample_per_dataset > 0:
            count = counts.get(asset.dataset, 0)
            if count >= sample_per_dataset and asset.sequence_id != "dataset":
                continue
            counts[asset.dataset] = count + 1
        selected.append(asset)
    return [head_check(asset, timeout_s=timeout_s) for asset in selected]


def head_check(asset: Asset, timeout_s: float = 15.0) -> LinkCheck:
    assert asset.url is not None
    if asset.extra.get("download_kind") == "egoexo_manifest":
        return s3_manifest_check(asset, timeout_s=timeout_s)
    request = urllib.request.Request(asset.url, method="HEAD", headers={"User-Agent": "gaze-pipeline/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return LinkCheck(
                dataset=asset.dataset,
                sequence_id=asset.sequence_id,
                asset_key=asset.asset_key,
                url=asset.url,
                ok=200 <= response.status < 400,
                status=response.status,
                content_type=response.headers.get("Content-Type"),
            )
    except urllib.error.HTTPError as exc:
        return LinkCheck(asset.dataset, asset.sequence_id, asset.asset_key, asset.url, False, exc.code, error=str(exc))
    except Exception as exc:  # Network portability matters more than typed exceptions here.
        return LinkCheck(asset.dataset, asset.sequence_id, asset.asset_key, asset.url, False, error=str(exc))


def fetch_assets(
    assets: list[Asset],
    raw_root: str | Path,
    dry_run: bool = False,
    workers: int = 1,
    asset_workers: int = 1,
    timeout_s: float = 120.0,
    progress_callback: ProgressCallback | None = None,
) -> list[dict]:
    root = Path(raw_root)
    asset_workers = max(1, asset_workers)

    def fetch_one(asset: Asset) -> dict:
        target = target_path_for_asset(root, asset)
        worker_count = max(1, workers)
        item = {
            "dataset": asset.dataset,
            "sequence_id": asset.sequence_id,
            "asset_key": asset.asset_key,
            "target": str(target),
            "dry_run": dry_run,
            "workers": worker_count,
            "asset_workers": asset_workers,
            "downloaded": False,
            "verified": False,
            "directory": asset.extra.get("download_kind") == "egoexo_manifest",
        }
        if dry_run:
            return item
        if asset.extra.get("download_kind") == "egoexo_manifest":
            try:
                item.update(download_egoexo_manifest_asset(asset, target, timeout_s=timeout_s, progress_callback=progress_callback))
                item["verified"] = True
            except Exception as exc:
                item["error"] = str(exc)
            return item
        if not asset.url:
            item["error"] = "asset has no URL; use the dataset provider instructions"
            return item
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            download_result = download_file(
                asset.url,
                target,
                expected_size=asset.size_bytes,
                workers=worker_count,
                timeout_s=timeout_s,
                progress_callback=progress_callback,
                progress_context={
                    "dataset": asset.dataset,
                    "sequence_id": asset.sequence_id,
                    "asset_key": asset.asset_key,
                    "target": str(target),
                },
            )
            item.update(download_result)
            if asset.sha1:
                actual = sha1_file(target)
                item["verified"] = actual == asset.sha1
                item["sha1"] = actual
                if actual != asset.sha1:
                    item["error"] = "sha1 mismatch"
            else:
                item["verified"] = True
        except Exception as exc:
            item["error"] = str(exc)
        return item

    if asset_workers == 1 or len(assets) <= 1:
        return [fetch_one(asset) for asset in assets]

    results: list[dict | None] = [None] * len(assets)
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(asset_workers, len(assets))) as executor:
        future_to_index = {executor.submit(fetch_one, asset): index for index, asset in enumerate(assets)}
        for future in concurrent.futures.as_completed(future_to_index):
            results[future_to_index[future]] = future.result()
    return [item for item in results if item is not None]


def target_path_for_asset(root: Path, asset: Asset) -> Path:
    if asset.extra.get("download_kind") == "egoexo_manifest":
        return root / asset.dataset / asset.sequence_id / asset.asset_key
    return root / asset.dataset / asset.sequence_id / asset.filename


def s3_manifest_check(asset: Asset, timeout_s: float = 15.0) -> LinkCheck:
    assert asset.url is not None
    credential_path = asset.extra.get("credential_path")
    if credential_path and not Path(credential_path).exists():
        return LinkCheck(asset.dataset, asset.sequence_id, asset.asset_key, asset.url, False, error=f"credential file not found: {credential_path}")
    try:
        completed = run_aws(asset, ["s3", "cp", asset.url, "-"], timeout_s=timeout_s, capture_output=True)
        ok = completed.returncode == 0 and completed.stdout.strip().startswith(b"[")
        return LinkCheck(
            dataset=asset.dataset,
            sequence_id=asset.sequence_id,
            asset_key=asset.asset_key,
            url=asset.url,
            ok=ok,
            status=0 if ok else completed.returncode,
            content_type="application/json" if ok else None,
            error=None if ok else completed.stderr.decode("utf-8", errors="replace")[-500:],
        )
    except Exception as exc:
        return LinkCheck(asset.dataset, asset.sequence_id, asset.asset_key, asset.url, False, error=str(exc))


def download_egoexo_manifest_asset(
    asset: Asset,
    target: Path,
    timeout_s: float,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    assert asset.url is not None
    target.mkdir(parents=True, exist_ok=True)
    manifest_bytes = run_aws(asset, ["s3", "cp", asset.url, "-"], timeout_s=timeout_s, capture_output=True, check=True).stdout
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    paths = select_egoexo_paths(manifest, asset.extra)
    downloaded = 0
    reused = 0
    bytes_total = sum(item.get("size") or 0 for item in paths)
    bytes_done = 0
    for item in paths:
        relative = Path(item["relative_path"])
        output = target / relative
        expected_size = item.get("size")
        if output.exists() and (expected_size is None or output.stat().st_size == expected_size):
            reused += 1
            bytes_done += expected_size or output.stat().st_size
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        if progress_callback:
            progress_callback(
                {
                    "dataset": asset.dataset,
                    "sequence_id": asset.sequence_id,
                    "asset_key": asset.asset_key,
                    "target": str(output),
                    "event": "download-start",
                    "bytes_done": bytes_done,
                    "bytes_total": bytes_total,
                }
            )
        run_aws(asset, ["s3", "cp", item["source_path"], str(output)], timeout_s=timeout_s, capture_output=True, check=True)
        if expected_size is not None and output.stat().st_size != expected_size:
            raise RuntimeError(f"downloaded size mismatch for {output}: {output.stat().st_size} != {expected_size}")
        downloaded += 1
        bytes_done += expected_size or output.stat().st_size
        if progress_callback:
            progress_callback(
                {
                    "dataset": asset.dataset,
                    "sequence_id": asset.sequence_id,
                    "asset_key": asset.asset_key,
                    "target": str(output),
                    "event": "download-progress",
                    "bytes_done": bytes_done,
                    "bytes_total": bytes_total,
                }
            )
    return {
        "downloaded": downloaded > 0,
        "reused": reused > 0,
        "files": len(paths),
        "files_downloaded": downloaded,
        "files_reused": reused,
        "bytes": bytes_done,
        "bytes_total": bytes_total,
        "range_download": False,
    }


def select_egoexo_paths(manifest: list[dict], filters: dict) -> list[dict]:
    benchmarks = set(filters.get("benchmarks") or [])
    views = set(filters.get("views") or [])
    splits = set(filters.get("splits") or [])
    universities = set(filters.get("universities") or [])
    selected: list[dict] = []
    for entry in manifest:
        entry_benchmarks = set(entry.get("benchmarks") or [])
        entry_splits = set(entry.get("splits") or [])
        if benchmarks and entry_benchmarks and not (entry_benchmarks & benchmarks):
            continue
        if splits and entry_splits and not (entry_splits & splits):
            continue
        for path in entry.get("paths") or []:
            path_views = set(path.get("views") or [])
            path_universities = set(path.get("universities") or [])
            if views and path_views and not (path_views & views):
                continue
            if universities and path_universities and not (path_universities & universities):
                continue
            selected.append(path)
    return selected


def run_aws(
    asset: Asset,
    args: list[str],
    timeout_s: float,
    capture_output: bool,
    check: bool = False,
) -> subprocess.CompletedProcess:
    env = aws_env_for_asset(asset)
    completed = subprocess.run(["aws", *args], env=env, timeout=timeout_s, capture_output=capture_output, check=False)
    if check and completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace") if completed.stderr else ""
        raise RuntimeError(f"aws {' '.join(args[:3])} failed with exit code {completed.returncode}: {stderr[-500:]}")
    return completed


def aws_env_for_asset(asset: Asset) -> dict[str, str]:
    env = os.environ.copy()
    credential_path = asset.extra.get("credential_path")
    if credential_path:
        credentials = parse_aws_credentials_file(credential_path)
        env.update(credentials)
    return env


def parse_aws_credentials_file(path: str | Path) -> dict[str, str]:
    source = Path(path)
    if not source.exists():
        return {}
    text = source.read_text(encoding="utf-8")
    values = {
        "AWS_ACCESS_KEY_ID": extract_credential_value(text, r"Access Key ID:\s*(\S+)"),
        "AWS_SECRET_ACCESS_KEY": extract_credential_value(text, r"Secret Access Key:\s*(\S+)"),
        "AWS_DEFAULT_REGION": extract_credential_value(text, r"Region:\s*(\S+)"),
    }
    return {key: value for key, value in values.items() if value}


def extract_credential_value(text: str, pattern: str) -> str | None:
    match = re.search(pattern, text)
    return match.group(1).strip() if match else None


def download_file(
    url: str,
    target: str | Path,
    expected_size: int | None = None,
    workers: int = 1,
    timeout_s: float = 120.0,
    progress_callback: ProgressCallback | None = None,
    progress_context: dict | None = None,
) -> dict:
    target = Path(target)
    workers = max(1, workers)
    context = progress_context or {}
    if target.exists() and (expected_size is None or target.stat().st_size == expected_size):
        return {
            "downloaded": False,
            "reused": True,
            "bytes": target.stat().st_size,
            "range_download": False,
        }
    if expected_size and workers > 1 and supports_range_request(url, timeout_s=timeout_s):
        return download_file_ranges(
            url,
            target,
            expected_size,
            workers=workers,
            timeout_s=timeout_s,
            progress_callback=progress_callback,
            progress_context=context,
        )
    return download_file_stream(
        url,
        target,
        expected_size=expected_size,
        timeout_s=timeout_s,
        progress_callback=progress_callback,
        progress_context=context,
    )


def supports_range_request(url: str, timeout_s: float = 120.0) -> bool:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "gaze-pipeline/0.1", "Range": "bytes=0-0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return response.status == 206
    except Exception:
        return False


def byte_ranges(size: int, workers: int) -> list[tuple[int, int]]:
    workers = max(1, min(workers, size))
    chunk_size = math.ceil(size / workers)
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < size:
        end = min(size - 1, start + chunk_size - 1)
        ranges.append((start, end))
        start = end + 1
    return ranges


def download_file_ranges(
    url: str,
    target: Path,
    expected_size: int,
    workers: int,
    timeout_s: float,
    progress_callback: ProgressCallback | None,
    progress_context: dict,
) -> dict:
    partial = target.with_suffix(target.suffix + ".part")
    marker_dir = target.with_suffix(target.suffix + ".ranges")
    marker_dir.mkdir(parents=True, exist_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("ab"):
        pass
    if partial.stat().st_size != expected_size:
        with partial.open("r+b") as handle:
            handle.truncate(expected_size)

    ranges = byte_ranges(expected_size, workers)
    completed_bytes = 0
    pending: list[tuple[int, int, int, Path]] = []
    for index, (start, end) in enumerate(ranges):
        marker = marker_dir / f"{index:05d}.done"
        length = end - start + 1
        if marker.exists():
            completed_bytes += length
        else:
            pending.append((index, start, end, marker))

    progress = {"bytes": completed_bytes}
    progress_lock = threading.Lock()
    if progress_callback:
        progress_callback(
            {
                **progress_context,
                "event": "download-start",
                "range_download": True,
                "workers": min(workers, len(ranges)),
                "bytes_done": completed_bytes,
                "bytes_total": expected_size,
            }
        )

    def fetch_range(start: int, end: int, marker: Path) -> None:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "gaze-pipeline/0.1", "Range": f"bytes={start}-{end}"},
        )
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            if response.status != 206:
                raise RuntimeError(f"range request returned HTTP {response.status}")
            with partial.open("r+b") as handle:
                handle.seek(start)
                while True:
                    chunk = response.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    with progress_lock:
                        progress["bytes"] += len(chunk)
        marker.write_text(json.dumps({"start": start, "end": end}), encoding="utf-8")

    started_at = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, max(1, len(pending)))) as executor:
        futures = [executor.submit(fetch_range, start, end, marker) for _, start, end, marker in pending]
        for future in concurrent.futures.as_completed(futures):
            future.result()
            if progress_callback:
                elapsed = max(0.001, time.monotonic() - started_at)
                with progress_lock:
                    bytes_done = progress["bytes"]
                progress_callback(
                    {
                        **progress_context,
                        "event": "download-progress",
                        "range_download": True,
                        "bytes_done": bytes_done,
                        "bytes_total": expected_size,
                        "elapsed_s": elapsed,
                    }
                )

    partial.replace(target)
    shutil.rmtree(marker_dir, ignore_errors=True)
    elapsed = max(0.001, time.monotonic() - started_at)
    return {
        "downloaded": True,
        "reused": False,
        "bytes": target.stat().st_size,
        "elapsed_s": elapsed,
        "range_download": True,
    }


def download_file_stream(
    url: str,
    target: Path,
    expected_size: int | None,
    timeout_s: float,
    progress_callback: ProgressCallback | None,
    progress_context: dict,
) -> dict:
    partial = target.with_suffix(target.suffix + ".part")
    headers = {"User-Agent": "gaze-pipeline/0.1"}
    existing = partial.stat().st_size if partial.exists() else 0
    if existing:
        headers["Range"] = f"bytes={existing}-"
    request = urllib.request.Request(url, headers=headers)
    started_at = time.monotonic()
    bytes_done = existing
    if progress_callback:
        progress_callback(
            {
                **progress_context,
                "event": "download-start",
                "range_download": False,
                "workers": 1,
                "bytes_done": bytes_done,
                "bytes_total": expected_size,
            }
        )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        if existing and response.status != 206:
            existing = 0
            bytes_done = 0
            mode = "wb"
        else:
            mode = "ab" if existing else "wb"
        with partial.open(mode) as handle:
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                bytes_done += len(chunk)
                if progress_callback:
                    progress_callback(
                        {
                            **progress_context,
                            "event": "download-progress",
                            "range_download": False,
                            "bytes_done": bytes_done,
                            "bytes_total": expected_size,
                            "elapsed_s": max(0.001, time.monotonic() - started_at),
                        }
                    )
    if expected_size is not None and partial.stat().st_size != expected_size:
        raise RuntimeError(f"downloaded size mismatch: {partial.stat().st_size} != {expected_size}")
    partial.replace(target)
    return {
        "downloaded": True,
        "reused": False,
        "bytes": target.stat().st_size,
        "elapsed_s": max(0.001, time.monotonic() - started_at),
        "range_download": False,
    }


def sha1_file(path: str | Path) -> str:
    digest = hashlib.sha1()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
