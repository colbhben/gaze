from __future__ import annotations

from dataclasses import asdict, dataclass
import bz2
import concurrent.futures
import gzip
import json
import lzma
from pathlib import Path
import posixpath
import shutil
import subprocess
import tarfile
import tempfile
from typing import Any, Callable
import zipfile

from .config import RectifyConfig, load_config
from .datasets import load_catalog
from .download import fetch_assets, plan_downloads, write_download_manifest
from .rectify import rectify_dataset
from .table import read_table, write_table


Runner = Callable[[list[str]], subprocess.CompletedProcess]
ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass
class S3Config:
    bucket_uri: str = "s3://far-research-internal/colbhben/gaze"
    upload_mode: str = "awscli"
    access_mode: str = "file_mount"
    mount_root: str = "/nfs"
    unprocessed_prefix: str = "unprocessed"
    processed_prefix: str = "processed"
    manifests_prefix: str = "manifests"
    splits_prefix: str = "splits"
    local_cache_root: str = ".gaze-cache"
    local_storage_reserve_bytes: int = 10_000_000_000
    local_storage_max_fraction: float = 0.9
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
    if cfg.upload_mode != "awscli":
        raise ValueError("upload_mode must be awscli")
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
    if not is_s3_uri(target_uri):
        raise ValueError("upload targets must be canonical s3:// URIs")
    return run_or_report(aws_cp_args(cfg, source, target_uri), dry_run=dry_run, runner=runner)


def upload_dir(cfg: S3Config, source: str | Path, target_uri: str, dry_run: bool = False, runner: Runner | None = None) -> dict[str, Any]:
    if not is_s3_uri(target_uri):
        raise ValueError("upload targets must be canonical s3:// URIs")
    return run_or_report(aws_sync_args(cfg, source, target_uri), dry_run=dry_run, runner=runner)


def s3_uri_for_asset(cfg: S3Config, asset: Any) -> str:
    if asset.extra.get("download_kind") == "egoexo_manifest":
        return cfg.unprocessed_uri(asset.dataset, asset.sequence_id, asset.asset_key)
    return join_s3_uri(cfg.unprocessed_uri(asset.dataset, asset.sequence_id, asset.asset_key), asset.filename)


def list_existing_s3_objects(cfg: S3Config, datasets: set[str], runner: Runner | None = None) -> dict[str, int]:
    return list_s3_objects(cfg, [cfg.unprocessed_uri(dataset) + "/" for dataset in sorted(datasets)], runner=runner)


def list_s3_objects(cfg: S3Config, prefixes: list[str], runner: Runner | None = None) -> dict[str, int]:
    objects: dict[str, int] = {}
    command_runner = runner
    if command_runner is None:
        command_runner = lambda command: subprocess.run(command, check=True, text=True, capture_output=True)
    for prefix in prefixes:
        completed = command_runner(aws_base_args(cfg) + ["ls", prefix, "--recursive"])
        for line in completed.stdout.splitlines():
            parts = line.split(maxsplit=3)
            if len(parts) != 4:
                continue
            try:
                size = int(parts[2])
            except ValueError:
                continue
            objects[parts[3]] = size
    return objects


def asset_exists_in_s3(cfg: S3Config, asset: Any, existing_objects: dict[str, int]) -> bool:
    uri = s3_uri_for_asset(cfg, asset)
    _, key = split_s3_uri(uri)
    size = existing_objects.get(key)
    if size is None:
        return False
    return asset.size_bytes is None or size == asset.size_bytes


def access_dir(cfg: S3Config, source: str | Path, target: str | Path, dry_run: bool = False, runner: Runner | None = None) -> dict[str, Any]:
    if cfg.access_mode == "file_mount":
        return sync_file_mount(cfg, source, target, dry_run=dry_run)
    return run_or_report(aws_sync_args(cfg, source, target), dry_run=dry_run, runner=runner)


def expand_archives_on_s3(
    cfg: S3Config,
    cache_root: str | Path | None = None,
    datasets: set[str] | None = None,
    dry_run: bool = False,
    runner: Runner | None = None,
    max_archives: int | None = None,
) -> dict[str, Any]:
    prefixes = [cfg.unprocessed_uri(dataset) + "/" for dataset in sorted(datasets)] if datasets else [cfg.unprocessed_uri() + "/"]
    existing_objects = list_s3_objects(cfg, prefixes, runner=runner)
    archives = [key for key in sorted(existing_objects) if is_archive_key(key)]
    if max_archives is not None:
        archives = archives[:max_archives]
    cache = Path(cache_root or Path(cfg.local_cache_root) / "archive-expansion")
    ensure_not_nfs_path(cfg, cache)
    operations = [
        expand_s3_archive(cfg, archive_key, existing_objects, cache, dry_run=dry_run, runner=runner)
        for archive_key in archives
    ]
    return {
        "layout_version": 1,
        "unprocessed_prefix": cfg.unprocessed_uri(),
        "datasets": sorted(datasets) if datasets else None,
        "archives_considered": len(archives),
        "expanded_archives": sum(1 for operation in operations if operation["uploaded_files"] or operation["skipped_existing_files"]),
        "uploaded_files": sum(operation["uploaded_files"] for operation in operations),
        "skipped_existing_files": sum(operation["skipped_existing_files"] for operation in operations),
        "operations": operations,
    }


def expand_s3_archive(
    cfg: S3Config,
    archive_key: str,
    existing_objects: dict[str, int],
    cache_root: Path,
    dry_run: bool = False,
    runner: Runner | None = None,
) -> dict[str, Any]:
    bucket, _ = split_s3_uri(cfg.bucket_uri)
    archive_uri = f"s3://{bucket}/{archive_key}"
    parent_key = archive_key.rsplit("/", 1)[0] if "/" in archive_key else ""
    operation: dict[str, Any] = {
        "archive_key": archive_key,
        "archive_uri": archive_uri,
        "source": None,
        "download": None,
        "members": 0,
        "uploaded_files": 0,
        "skipped_existing_files": 0,
        "uploads": [],
        "skips": [],
        "errors": [],
        "dry_run": dry_run,
    }
    source_path = archive_source_path(cfg, archive_uri, cache_root, dry_run=dry_run, runner=runner, operation=operation)
    if source_path is None:
        return operation
    archive_stage = cache_root / "expanded" / safe_cache_name(archive_key)
    if not dry_run:
        archive_stage.mkdir(parents=True, exist_ok=True)
    try:
        for member in iter_archive_members(source_path):
            operation["members"] += 1
            target_key = join_s3_key(parent_key, member["name"])
            expected_size = member.get("size")
            existing_size = existing_objects.get(target_key)
            if existing_size is not None and (expected_size is None or existing_size == expected_size):
                operation["skipped_existing_files"] += 1
                operation["skips"].append({"key": target_key, "reason": "already present", "size_bytes": existing_size})
                continue
            target_uri = f"s3://{bucket}/{target_key}"
            if dry_run:
                operation["uploads"].append({"source_member": member["name"], "target_uri": target_uri, "dry_run": True})
                continue
            staged = archive_stage / member["name"]
            staged.parent.mkdir(parents=True, exist_ok=True)
            member["extract"](staged)
            upload_result = upload_file(cfg, staged, target_uri, dry_run=False, runner=runner)
            operation["uploads"].append({"source_member": member["name"], "target_uri": target_uri, "upload": upload_result})
            operation["uploaded_files"] += 1
            existing_objects[target_key] = staged.stat().st_size
            staged.unlink(missing_ok=True)
    except Exception as exc:
        operation["errors"].append(str(exc))
    finally:
        if not dry_run:
            shutil.rmtree(archive_stage, ignore_errors=True)
    return operation


def archive_source_path(
    cfg: S3Config,
    archive_uri: str,
    cache_root: Path,
    dry_run: bool,
    runner: Runner | None,
    operation: dict[str, Any],
) -> Path | None:
    if cfg.access_mode == "file_mount":
        mounted = mount_path_for_uri(cfg, archive_uri)
        if mounted.exists():
            operation["source"] = {"transport": "file_mount", "path": str(mounted)}
            return mounted
    local_archive = cache_root / "archives" / safe_cache_name(split_s3_uri(archive_uri)[1])
    operation["source"] = {"transport": "awscli", "path": str(local_archive)}
    if dry_run:
        return None
    local_archive.parent.mkdir(parents=True, exist_ok=True)
    operation["download"] = run_or_report(aws_cp_args(cfg, archive_uri, local_archive), dry_run=False, runner=runner)
    return local_archive


def iter_archive_members(path: Path):
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                name = safe_member_name(info.filename)
                if name is None:
                    continue

                def extract_zip(target: Path, archive=archive, info=info) -> None:
                    with archive.open(info) as source, target.open("wb") as output:
                        shutil.copyfileobj(source, output)

                yield {"name": name, "size": info.file_size, "extract": extract_zip}
        return
    if tarfile.is_tarfile(path):
        with tarfile.open(path) as archive:
            for info in archive.getmembers():
                if not info.isfile():
                    continue
                name = safe_member_name(info.name)
                if name is None:
                    continue

                def extract_tar(target: Path, archive=archive, info=info) -> None:
                    source = archive.extractfile(info)
                    if source is None:
                        raise ValueError(f"unable to extract {info.name}")
                    with source, target.open("wb") as output:
                        shutil.copyfileobj(source, output)

                yield {"name": name, "size": info.size, "extract": extract_tar}
        return
    single_name = single_file_archive_member_name(path)
    if single_name is None:
        return

    def extract_single(target: Path) -> None:
        opener = opener_for_compressed_file(path)
        with opener(path, "rb") as source, target.open("wb") as output:
            shutil.copyfileobj(source, output)

    yield {"name": single_name, "size": None, "extract": extract_single}


def is_archive_key(key: str) -> bool:
    lowered = key.lower()
    return lowered.endswith((".zip", ".tar", ".tar.gz", ".tgz", ".gz", ".bz2", ".xz"))


def safe_member_name(name: str) -> str | None:
    normalized = posixpath.normpath(name.replace("\\", "/"))
    if normalized in {"", "."} or normalized.startswith("../") or normalized.startswith("/"):
        return None
    return normalized


def safe_cache_name(key: str) -> str:
    return key.replace("/", "__").replace(":", "_")


def join_s3_key(parent: str, member: str) -> str:
    return f"{parent.strip('/')}/{member.strip('/')}" if parent else member.strip("/")


def single_file_archive_member_name(path: Path) -> str | None:
    name = path.name
    lowered = name.lower()
    for suffix in (".gz", ".bz2", ".xz"):
        if lowered.endswith(suffix) and not lowered.endswith(".tar" + suffix):
            return name[: -len(suffix)]
    return None


def opener_for_compressed_file(path: Path):
    lowered = path.name.lower()
    if lowered.endswith(".gz"):
        return gzip.open
    if lowered.endswith(".bz2"):
        return bz2.open
    if lowered.endswith(".xz"):
        return lzma.open
    raise ValueError(f"unsupported compressed file: {path}")


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
    max_download_bytes: int | None = None,
    reserve_bytes: int | None = None,
    storage_fraction: float | None = None,
    include_unknown_size: bool = False,
    clean_after_upload: bool = True,
    workers: int = 1,
    dataset_workers: dict[str, int] | None = None,
    download_timeout_s: float = 120.0,
    progress_callback: ProgressCallback | None = None,
    stream_uploads: bool = False,
    asset_workers: int = 1,
    assets_override: list[Any] | None = None,
    skip_existing_s3: bool = False,
    interleave_datasets: bool = False,
) -> dict[str, Any]:
    if assets_override is None:
        catalog = load_catalog(repo_root)
        assets = plan_downloads(catalog, datasets=datasets, modalities=modalities, sequences=sequences)
    else:
        assets = assets_override
    existing_s3_objects: dict[str, int] = {}
    skipped_existing_s3 = []
    if skip_existing_s3 and assets:
        existing_s3_objects = list_existing_s3_objects(cfg, {asset.dataset for asset in assets}, runner=runner)
        remaining_assets = []
        for asset in assets:
            if asset_exists_in_s3(cfg, asset, existing_s3_objects):
                skipped_existing_s3.append(asset)
            else:
                remaining_assets.append(asset)
        assets = remaining_assets
    raw_root = Path(raw_root)
    ensure_not_nfs_path(cfg, raw_root)
    storage = storage_budget(
        raw_root,
        reserve_bytes=cfg.local_storage_reserve_bytes if reserve_bytes is None else reserve_bytes,
        storage_fraction=cfg.local_storage_max_fraction if storage_fraction is None else storage_fraction,
        max_download_bytes=max_download_bytes,
    )
    selection = select_assets_for_storage(assets, storage["budget_bytes"], include_unknown_size=include_unknown_size)
    assets = selection["selected"]
    if interleave_datasets:
        assets = interleave_assets_by_dataset(assets)
    download_manifest = raw_root / "download_manifest.json"
    if not dry_run:
        write_download_manifest(assets, download_manifest)
    operations: list[dict[str, Any] | None] = [None] * len(assets)
    upload_runner = runner
    if stream_uploads and upload_runner is None:
        upload_runner = lambda command: subprocess.run(command, check=True, text=True)

    def process_asset(index: int, asset: Any) -> dict[str, Any]:
        worker_count = (dataset_workers or {}).get(asset.dataset, workers)
        fetch_result = fetch_assets(
            [asset],
            raw_root,
            dry_run=dry_run,
            workers=worker_count,
            asset_workers=1,
            timeout_s=download_timeout_s,
            progress_callback=progress_callback,
        )[0]
        target = Path(fetch_result["target"])
        if fetch_result.get("error"):
            s3_target = cfg.unprocessed_uri(asset.dataset, asset.sequence_id, asset.asset_key)
            return {"asset": asdict(asset), "fetch": fetch_result, "upload": {"skipped": True, "reason": "fetch failed"}, "s3_uri": s3_target}
        if fetch_result.get("directory"):
            s3_target = cfg.unprocessed_uri(asset.dataset, asset.sequence_id, asset.asset_key)
            upload_result = upload_dir(cfg, target, s3_target, dry_run=dry_run, runner=upload_runner)
            if clean_after_upload and not dry_run and upload_result.get("returncode") == 0 and target.exists():
                shutil.rmtree(target, ignore_errors=True)
        else:
            s3_target = join_s3_uri(cfg.unprocessed_uri(asset.dataset, asset.sequence_id, asset.asset_key), asset.filename)
            upload_result = upload_file(cfg, target, s3_target, dry_run=dry_run, runner=upload_runner)
            if clean_after_upload and not dry_run and upload_result.get("returncode") == 0 and target.exists():
                target.unlink()
        return {"asset": asdict(asset), "fetch": fetch_result, "upload": upload_result, "s3_uri": s3_target}

    asset_workers = max(1, asset_workers)
    if asset_workers == 1 or len(assets) <= 1:
        for index, asset in enumerate(assets):
            operations[index] = process_asset(index, asset)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(asset_workers, len(assets))) as executor:
            future_to_index = {executor.submit(process_asset, index, asset): index for index, asset in enumerate(assets)}
            for future in concurrent.futures.as_completed(future_to_index):
                operations[future_to_index[future]] = future.result()
    manifest_upload = None
    if dry_run:
        manifest_upload = {"s3_uri": cfg.download_manifest_uri(manifest_name), "dry_run": True}
    else:
        manifest_upload = upload_file(cfg, download_manifest, cfg.download_manifest_uri(manifest_name), dry_run=False, runner=runner)
    return {
        "layout_version": 1,
        "storage": storage,
        "selection": {
            "selected_assets": len(selection["selected"]),
            "selected_bytes": selection["selected_bytes"],
            "skipped_assets": [asdict(asset) for asset in selection["skipped"]],
            "skipped_existing_s3_assets": [asdict(asset) for asset in skipped_existing_s3],
            "skip_existing_s3": skip_existing_s3,
            "interleave_datasets": interleave_datasets,
        },
        "unprocessed_prefix": cfg.unprocessed_uri(),
        "manifest_uri": cfg.download_manifest_uri(manifest_name),
        "operations": [operation for operation in operations if operation is not None],
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
        ensure_not_nfs_path(cfg, partition_raw)
        ensure_not_nfs_path(cfg, partition_canonical)
        download = access_dir(cfg, cfg.unprocessed_uri(dataset, partition), partition_raw, dry_run=dry_run, runner=runner)
        rows: list[dict[str, Any]] = []
        if not dry_run:
            rows = rectify_dataset(raw_cache, partition_canonical, config=config, dataset=dataset, episodes={partition})
            manifest_rows.extend(add_s3_uris(rows, cfg, profile))
        episode_uploads = []
        for row in rows:
            episode_path = partition_canonical / row["episode_path"]
            target_uri = cfg.processed_uri(profile, row["dataset"], row["episode_id"])
            episode_uploads.append(upload_dir(cfg, episode_path, target_uri, dry_run=dry_run, runner=runner))
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
            s3_uri = cfg.processed_uri(profile_name, dataset, episode_id)
            episodes.append(
                {
                    "split": split_name,
                    "id": episode,
                    "dataset": dataset,
                    "episode_id": episode_id,
                    "s3_uri": s3_uri,
                    "nfs_path": str(mount_path_for_uri(cfg, s3_uri)) if cfg.access_mode == "file_mount" else None,
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
        operations.append(access_dir(cfg, episode["s3_uri"], target, dry_run=dry_run, runner=runner))
    return {"episodes": [op for op in manifest.get("episodes", []) if not split or op["split"] == split], "operations": operations}


def ensure_not_nfs_path(cfg: S3Config, path: str | Path) -> None:
    resolved = Path(path).resolve(strict=False)
    mount = Path(cfg.mount_root).resolve(strict=False)
    if resolved == mount or mount in resolved.parents:
        raise ValueError(f"local working root must not be under the NFS mount {cfg.mount_root}: {path}")


def storage_budget(
    root: str | Path,
    reserve_bytes: int,
    storage_fraction: float,
    max_download_bytes: int | None = None,
) -> dict[str, Any]:
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    fraction_budget = int(usage.free * storage_fraction)
    reserve_budget = max(0, usage.free - reserve_bytes)
    budget = min(fraction_budget, reserve_budget)
    if max_download_bytes is not None:
        budget = min(budget, max_download_bytes)
    return {
        "root": str(path),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "reserve_bytes": reserve_bytes,
        "storage_fraction": storage_fraction,
        "budget_bytes": max(0, budget),
    }


def select_assets_for_storage(assets: list, budget_bytes: int, include_unknown_size: bool = False) -> dict[str, Any]:
    known = [asset for asset in assets if asset.size_bytes is not None]
    unknown = [asset for asset in assets if asset.size_bytes is None]
    ordered = sorted(known, key=lambda asset: asset.size_bytes or 0, reverse=True)
    selected = []
    skipped = []
    used = 0
    for asset in ordered:
        size = int(asset.size_bytes or 0)
        if used + size <= budget_bytes:
            selected.append(asset)
            used += size
        else:
            skipped.append(asset)
    if include_unknown_size:
        selected.extend(unknown)
    else:
        skipped.extend(unknown)
    return {"selected": selected, "skipped": skipped, "selected_bytes": used}


def interleave_assets_by_dataset(assets: list) -> list:
    buckets: dict[str, list] = {}
    for asset in assets:
        buckets.setdefault(asset.dataset, []).append(asset)
    interleaved = []
    while buckets:
        for dataset in list(buckets):
            bucket = buckets[dataset]
            interleaved.append(bucket.pop(0))
            if not bucket:
                del buckets[dataset]
    return interleaved


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
