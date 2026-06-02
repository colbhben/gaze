from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import urllib.error
import urllib.request

from .datasets import Asset, DatasetCatalog, filter_assets, summarize_assets


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


def fetch_assets(assets: list[Asset], raw_root: str | Path, dry_run: bool = False) -> list[dict]:
    root = Path(raw_root)
    results = []
    for asset in assets:
        target = root / asset.dataset / asset.sequence_id / asset.filename
        item = {
            "dataset": asset.dataset,
            "sequence_id": asset.sequence_id,
            "asset_key": asset.asset_key,
            "target": str(target),
            "dry_run": dry_run,
            "downloaded": False,
            "verified": False,
        }
        if dry_run:
            results.append(item)
            continue
        if not asset.url:
            item["error"] = "asset has no URL; use the dataset provider instructions"
            results.append(item)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(target.suffix + ".part")
        headers = {"User-Agent": "gaze-pipeline/0.1"}
        existing = partial.stat().st_size if partial.exists() else 0
        if existing:
            headers["Range"] = f"bytes={existing}-"
        request = urllib.request.Request(asset.url, headers=headers)
        try:
            with urllib.request.urlopen(request) as response, partial.open("ab" if existing else "wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
            partial.replace(target)
            item["downloaded"] = True
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
        results.append(item)
    return results


def sha1_file(path: str | Path) -> str:
    digest = hashlib.sha1()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
