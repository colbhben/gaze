from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Callable

from .config import RectifyConfig, load_config
from .datasets import load_catalog
from .download import fetch_assets, plan_downloads, write_download_manifest
from .rectify import rectify_dataset
from .table import read_table, write_table


Runner = Callable[[list[str]], subprocess.CompletedProcess]


@dataclass
class S3Config:
    bucket_uri: str = "s3://far-research-internal/colbhben/gaze"
    access_mode: str = "file_mount"
    mount_root: str = "/nfs"
    unprocessed_prefix: str = "unprocessed"
    processed_prefix: str = "processed"
    manifests_prefix: str = "manifests"
    splits_prefix: str = "splits"
    local_cache_root: str = ".gaze-cache"
    profile_name: str = "default-10hz"
    aws_profile: str | None = None
    aws_region: str | None = None
    storage_class: str | None = None
    server_side_encryption: str | None = None

    def base_uri(self) -> str:
        return self.bucket_uri.rstrip("/")

    def unprocessed_uri(self, dataset: str | None = None, partition: str | None = None, asset_key: str | None = None) -> str:
        return join_s3_uri(self.base_uri(), self.unprocessed_prefix, dataset, partition, asset_key)

    def processed_uri(self, profile: str | None = None, dataset: str | None = None, episode_id: str | None = None) -> str:
        return join_s3_uri(self.base_uri(), self.processed_prefix, profile or self.profile_name, dataset, episode_id)

    def processed_manifest_uri(self, profile: str | None = None) -> str:
        return join_s3_uri(self.processed_uri(profile), "manifest.jsonl")

    def download_manifest_uri(self, name: str) -> str:
        return join_s3_uri(self.base_uri(), self.manifests_prefix, "downloads", f"{name}.json")

    def split_uri(self, name: str) -> str:
        return join_s3_uri(self.base_uri(), self.splits_prefix, f"{name}.s3.json")


def load_s3_config(path: str | Path | None = None) -> S3Config:
    config_path = Path(path) if path else None
    data = json.loads(config_path.read_text(encoding="utf-8")) if config_path and config_path.exists() else {}
    cfg = S3Config(**data)
    if not cfg.bucket_uri.startswith("s3://"):
        raise ValueError("bucket_uri must start with s3://")
    if cfg.access_mode not in {"file_mount", "awscli"}:
        raise ValueError("access_mode must be file_mount or awscli")
    return cfg


def join_s3_uri(*parts: str | None) -> str:
    cleaned = [part.strip("/") for part in parts if part]
    if not cleaned:
        raise ValueError("cannot build empty S3 URI")
    first = cleaned[0]
    if first.startswith("s3://"):
        return first.rstrip("/") + ("/" + "/".join(cleaned[1:]) if len(cleaned) > 1 else "")
    return "/".join(cleaned)


def aws_base_args(cfg: S3Config) -> list[str]:
    args = ["aws"]
    if cfg.aws_profile:
        args.extend(["--profile", cfg.aws_profile])
    if cfg.aws_region:
        args.extend(["--region", cfg.aws_region])
    args.append("s3")
    return args


def aws_cp_args(cfg: S3Config, source: str | Path, target: str | Path, recursive: bool = False) -> list[str]:
    args = aws_base_args(cfg) + ["cp", str(source), str(target)]
    if recursive:
        args.append("--recursive")
    if cfg.storage_class:
        args.extend(["--storage-class", cfg.storage_class])
    if cfg.server_side_encryption:
        args.extend(["--sse", cfg.server_side_encryption])
    return args


def aws_sync_args(cfg: S3Config, source: str | Path, target: str | Path) -> list[str]:
    args = aws_base_args(cfg) + ["sync", str(source), str(target)]
    if cfg.storage_class:
        args.extend(["--storage-class", cfg.storage_class])
    if cfg.server_side_encryption:
        args.extend(["--sse", cfg.server_side_encryption])
    return args


def is_s3_uri(value: str | Path) -> bool:
    return str(value).startswith("s3://")


def split_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"not an S3 URI: {uri}")
    remainder = uri[len("s3://") :]
    bucket, _, key = remainder.partition("/")
    return bucket, key


def mount_path_for_uri(cfg: S3Config, uri: str) -> Path:
    bucket, key = split_s3_uri(uri)
    configured_bucket, _ = split_s3_uri(cfg.bucket_uri)
    if bucket != configured_bucket:
        raise ValueError(f"mounted bucket mismatch: {bucket} != {configured_bucket}")
    return Path(cfg.mount_root) / key


def storage_path(cfg: S3Config, value: str | Path) -> Path:
    text = str(value)
    return mount_path_for_uri(cfg, text) if is_s3_uri(text) else Path(value)


def run_or_report(args: list[str], dry_run: bool, runner: Runner | None = None) -> dict[str, Any]:
    item: dict[str, Any] = {"transport": "awscli", "command": args, "dry_run": dry_run}
    if dry_run:
        return item
    if runner is None:
        runner = lambda command: subprocess.run(command, check=True, text=True, capture_output=True)
    completed = runner(args)
    item["returncode"] = completed.returncode
    item["stdout"] = completed.stdout
    item["stderr"] = completed.stderr
    return item


def upload_file(cfg: S3Config, source: Path, target_uri: str, dry_run: bool = False, runner: Runner | None = None) -> dict[str, Any]:
    if cfg.access_mode == "file_mount":
        return copy_file_mount(cfg, source, target_uri, dry_run=dry_run)
    return run_or_report(aws_cp_args(cfg, source, target_uri), dry_run=dry_run, runner=runner)


def sync_dir(cfg: S3Config, source: str | Path, target: str | Path, dry_run: bool = False, runner: Runner | None = None) -> dict[str, Any]:
    if cfg.access_mode == "file_mount":
        return sync_file_mount(cfg, source, target, dry_run=dry_run)
    return run_or_report(aws_sync_args(cfg, source, target), dry_run=dry_run, runner=runner)


def copy_file_mount(cfg: S3Config, source: str | Path, target: str | Path, dry_run: bool = False) -> dict[str, Any]:
    source_path = storage_path(cfg, source)
    target_path = storage_path(cfg, target)
    item = {
        "transport": "file_mount",
        "operation": "copy",
        "source": str(source),
        "target": str(target),
        "source_path": str(source_path),
        "target_path": str(target_path),
        "dry_run": dry_run,
    }
    if dry_run:
        return item
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)
    item["copied"] = True
    return item


def sync_file_mount(cfg: S3Config, source: str | Path, target: str | Path, dry_run: bool = False) -> dict[str, Any]:
    source_path = storage_path(cfg, source)
    target_path = storage_path(cfg, target)
    item = {
        "transport": "file_mount",
        "operation": "sync",
        "source": str(source),
        "target": str(target),
        "source_path": str(source_path),
        "target_path": str(target_path),
        "dry_run": dry_run,
    }
    if dry_run:
        return item
    if source_path.is_file():
        target_path.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path / source_path.name)
        item["files_copied"] = 1
        return item
    target_path.mkdir(parents=True, exist_ok=True)
    files_copied = 0
    for path in source_path.rglob("*"):
        if path.is_dir():
            continue
        relative = path.relative_to(source_path)
        destination = target_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        files_copied += 1
    item["files_copied"] = files_copied
    return item


def serial_download_backup(
    repo_root: str | Path,
    cfg: S3Config,
    raw_root: str | Path,
    datasets: set[str] | None = None,
    modalities: set[str] | None = None,
    sequences: set[str] | None = None,
    dry_run: bool = False,
    runner: Runner | None = None,
    manifest_name: str = "latest",
) -> dict[str, Any]:
    catalog = load_catalog(repo_root)
    assets = plan_downloads(catalog, datasets=datasets, modalities=modalities, sequences=sequences)
    raw_root = Path(raw_root)
    download_manifest = raw_root / "download_manifest.json"
    if not dry_run:
        write_download_manifest(assets, download_manifest)
    operations = []
    for asset in assets:
        fetch_result = fetch_assets([asset], raw_root, dry_run=dry_run)[0]
        target = Path(fetch_result["target"])
        s3_target = join_s3_uri(cfg.unprocessed_uri(asset.dataset, asset.sequence_id, asset.asset_key), asset.filename)
        upload_result = upload_file(cfg, target, s3_target, dry_run=dry_run, runner=runner)
        operations.append({"asset": asdict(asset), "fetch": fetch_result, "upload": upload_result, "s3_uri": s3_target})
    manifest_upload = None
    if dry_run:
        manifest_upload = {"s3_uri": cfg.download_manifest_uri(manifest_name), "dry_run": True}
    else:
        manifest_upload = upload_file(cfg, download_manifest, cfg.download_manifest_uri(manifest_name), dry_run=False, runner=runner)
    return {
        "layout_version": 1,
        "unprocessed_prefix": cfg.unprocessed_uri(),
        "manifest_uri": cfg.download_manifest_uri(manifest_name),
        "operations": operations,
        "manifest_upload": manifest_upload,
    }


def serial_process_backup(
    cfg: S3Config,
    partitions: list[tuple[str, str]],
    config: RectifyConfig | None = None,
    local_cache_root: str | Path | None = None,
    dry_run: bool = False,
    runner: Runner | None = None,
    clean: bool = True,
) -> dict[str, Any]:
    cache = Path(local_cache_root or cfg.local_cache_root)
    profile = (config.profile_name if config else cfg.profile_name) or cfg.profile_name
    raw_cache = cache / "unprocessed"
    processed_cache = cache / "processed"
    manifest_rows: list[dict[str, Any]] = []
    operations = []
    for dataset, partition in partitions:
        partition_raw = raw_cache / dataset / partition
        partition_canonical = processed_cache / profile
        download = sync_dir(cfg, cfg.unprocessed_uri(dataset, partition), partition_raw, dry_run=dry_run, runner=runner)
        rows: list[dict[str, Any]] = []
        if not dry_run:
            rows = rectify_dataset(raw_cache, partition_canonical, config=config, dataset=dataset, episodes={partition})
            manifest_rows.extend(add_s3_uris(rows, cfg, profile))
        episode_uploads = []
        for row in rows:
            episode_path = partition_canonical / row["episode_path"]
            target_uri = cfg.processed_uri(profile, row["dataset"], row["episode_id"])
            episode_uploads.append(sync_dir(cfg, episode_path, target_uri, dry_run=dry_run, runner=runner))
        operations.append({"dataset": dataset, "partition": partition, "download": download, "episodes": rows, "uploads": episode_uploads})
        if clean and not dry_run:
            shutil.rmtree(partition_raw, ignore_errors=True)
            for row in rows:
                shutil.rmtree(partition_canonical / row["episode_path"], ignore_errors=True)
    manifest_dir = processed_cache / profile
    canonical_manifest_path = manifest_dir / "manifest.parquet"
    export_manifest_path = manifest_dir / "manifest.jsonl"
    manifest_upload = None
    if dry_run:
        manifest_upload = {"s3_uri": cfg.processed_manifest_uri(profile), "dry_run": True}
    else:
        canonical_manifest_output = write_table(manifest_rows, canonical_manifest_path)
        write_table(manifest_rows, export_manifest_path)
        manifest_upload = upload_file(cfg, export_manifest_path, cfg.processed_manifest_uri(profile), dry_run=False, runner=runner)
    return {
        "layout_version": 1,
        "processed_prefix": cfg.processed_uri(profile),
        "manifest_uri": cfg.processed_manifest_uri(profile),
        "local_manifest": str(canonical_manifest_output) if not dry_run else str(canonical_manifest_path),
        "operations": operations,
        "manifest_upload": manifest_upload,
    }


def add_s3_uris(rows: list[dict[str, Any]], cfg: S3Config, profile: str) -> list[dict[str, Any]]:
    enriched = []
    for row in rows:
        item = dict(row)
        item["s3_uri"] = cfg.processed_uri(profile, row["dataset"], row["episode_id"])
        enriched.append(item)
    return enriched


def create_s3_pull_manifest(
    cfg: S3Config,
    split_path: str | Path,
    output_path: str | Path,
    profile: str | None = None,
    upload: bool = False,
    dry_run: bool = False,
    runner: Runner | None = None,
) -> dict[str, Any]:
    split = json.loads(Path(split_path).read_text(encoding="utf-8"))
    profile_name = profile or cfg.profile_name
    episodes = []
    for split_name, ids in split.get("splits", {}).items():
        for episode in ids:
            dataset, episode_id = episode.split(":", 1)
            episodes.append(
                {
                    "split": split_name,
                    "id": episode,
                    "dataset": dataset,
                    "episode_id": episode_id,
                    "s3_uri": cfg.processed_uri(profile_name, dataset, episode_id),
                }
            )
    manifest = {
        "layout_version": 1,
        "profile": profile_name,
        "source_split": Path(split_path).name,
        "processed_prefix": cfg.processed_uri(profile_name),
        "episodes": episodes,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    if upload:
        manifest["upload"] = upload_file(cfg, output, cfg.split_uri(output.stem), dry_run=dry_run, runner=runner)
    return manifest


def pull_processed_from_manifest(
    cfg: S3Config,
    pull_manifest: str | Path,
    dest_root: str | Path,
    split: str | None = None,
    dry_run: bool = False,
    runner: Runner | None = None,
) -> dict[str, Any]:
    manifest = json.loads(Path(pull_manifest).read_text(encoding="utf-8"))
    dest = Path(dest_root)
    operations = []
    for episode in manifest.get("episodes", []):
        if split and episode["split"] != split:
            continue
        target = dest / "episodes" / episode["dataset"] / episode["episode_id"]
        operations.append(sync_dir(cfg, episode["s3_uri"], target, dry_run=dry_run, runner=runner))
    return {"episodes": [op for op in manifest.get("episodes", []) if not split or op["split"] == split], "operations": operations}


def parse_partitions(value: str) -> list[tuple[str, str]]:
    partitions = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError("partitions must be dataset:partition_id entries")
        dataset, partition = item.split(":", 1)
        partitions.append((dataset, partition))
    return partitions


def config_from_path(path: str | Path | None) -> RectifyConfig | None:
    return load_config(path) if path else None
