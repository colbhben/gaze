from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

from . import __version__
from .config import load_config
from .datasets import load_catalog
from .download import estimate_downloads, fetch_assets, plan_downloads, verify_links, write_download_manifest
from .rectify import rectify_dataset
from .s3 import (
    create_s3_pull_manifest,
    load_s3_config,
    mount_path_for_uri,
    parse_partitions,
    pull_processed_from_manifest,
    serial_download_backup,
    serial_process_backup,
)
from .server import serve
from .splits import SplitRequest, create_split
from .table import parquet_available
from .validate import validate_canonical_root


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    try:
        return args.func(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gaze", description="Gaze dataset planning, rectification, validation, and viewing.")
    parser.add_argument("--version", action="version", version=f"gaze {__version__}")
    sub = parser.add_subparsers()

    doctor = sub.add_parser("doctor", help="Check local runtime dependencies.")
    doctor.set_defaults(func=cmd_doctor)

    datasets = sub.add_parser("datasets", help="Dataset catalog and download commands.")
    datasets_sub = datasets.add_subparsers(required=True)
    add_dataset_common(datasets_sub.add_parser("plan", help="Estimate selected dataset assets.")).set_defaults(func=cmd_datasets_plan)
    verify = add_dataset_common(datasets_sub.add_parser("verify-links", help="HEAD-check docs and sampled manifest links."))
    verify.add_argument("--sample-per-dataset", type=int, default=3)
    verify.add_argument("--timeout", type=float, default=15.0)
    verify.set_defaults(func=cmd_datasets_verify)
    fetch = add_dataset_common(datasets_sub.add_parser("fetch", help="Fetch selected assets explicitly."))
    fetch.add_argument("--raw-root", required=True)
    fetch.add_argument("--dry-run", action="store_true")
    fetch.add_argument("--manifest-out")
    fetch.set_defaults(func=cmd_datasets_fetch)

    rectify = sub.add_parser("rectify", help="Rectify raw fixture/dataset roots into canonical episodes.")
    rectify.add_argument("--raw-root", required=True)
    rectify.add_argument("--canonical-root", required=True)
    rectify.add_argument("--dataset")
    rectify.add_argument("--episodes", help="Comma-separated episode ids")
    rectify.add_argument("--config")
    rectify.add_argument("--set", action="append", default=[], dest="overrides", help="Config override, e.g. target_hz=5")
    rectify.set_defaults(func=cmd_rectify)

    validate = sub.add_parser("validate", help="Validation commands.")
    validate_sub = validate.add_subparsers(required=True)
    alignment = validate_sub.add_parser("alignment", help="Validate raw-to-canonical alignment.")
    alignment.add_argument("--canonical-root", required=True)
    alignment.add_argument("--raw-root")
    alignment.set_defaults(func=cmd_validate_alignment)

    split = sub.add_parser("split", help="Split commands.")
    split_sub = split.add_subparsers(required=True)
    create = split_sub.add_parser("create", help="Create seeded train/holdout manifests.")
    create.add_argument("--canonical-root", required=True)
    create.add_argument("--name", default="default")
    create.add_argument("--ratios", default="train=0.8,holdout=0.2")
    create.add_argument("--seed", type=int, default=0)
    create.add_argument("--mode", choices=["heterogeneous", "homogeneous"], default="heterogeneous")
    create.add_argument("--include-datasets")
    create.add_argument("--include-modalities")
    create.add_argument("--group-by", default="dataset")
    create.add_argument("--stratify-by")
    create.set_defaults(func=cmd_split_create)

    serve_parser = sub.add_parser("serve", help="Serve the local API and browser viewer.")
    add_serve_args(serve_parser)
    serve_parser.set_defaults(func=cmd_serve)

    view = sub.add_parser("view", help="Launch the local browser viewer.")
    add_serve_args(view)
    view.set_defaults(open=True, func=cmd_serve)

    s3 = sub.add_parser("s3", help="S3-backed serial pipeline commands.")
    s3_sub = s3.add_subparsers(required=True)
    layout = s3_sub.add_parser("layout", help="Show the configured static S3 layout.")
    add_s3_config_arg(layout)
    layout.set_defaults(func=cmd_s3_layout)

    backup_raw = add_dataset_common(s3_sub.add_parser("backup-raw", help="Serially download selected assets and back them up to S3."))
    add_s3_config_arg(backup_raw)
    backup_raw.add_argument("--raw-root", default=".gaze-cache/raw")
    backup_raw.add_argument("--manifest-name", default="latest")
    backup_raw.add_argument("--max-download-bytes", type=int, help="Optional cap for one local download batch")
    backup_raw.add_argument("--reserve-bytes", type=int, help="Local free-space reserve to keep unused")
    backup_raw.add_argument("--storage-fraction", type=float, help="Maximum fraction of currently free local space to use")
    backup_raw.add_argument("--include-unknown-size", action="store_true", help="Allow assets without known sizes into the batch")
    backup_raw.add_argument("--keep-cache", action="store_true", help="Keep local downloaded files after successful upload")
    backup_raw.add_argument("--dry-run", action="store_true")
    backup_raw.set_defaults(func=cmd_s3_backup_raw)

    process_serial = s3_sub.add_parser("process-serial", help="Serially pull raw partitions from S3, rectify, and upload processed episodes.")
    add_s3_config_arg(process_serial)
    process_serial.add_argument("--partitions", required=True, help="Comma-separated dataset:partition entries")
    process_serial.add_argument("--local-cache-root")
    process_serial.add_argument("--config")
    process_serial.add_argument("--dry-run", action="store_true")
    process_serial.add_argument("--keep-cache", action="store_true")
    process_serial.set_defaults(func=cmd_s3_process_serial)

    pull_manifest = s3_sub.add_parser("create-pull-manifest", help="Create a static S3 pull manifest from a train/holdout split.")
    add_s3_config_arg(pull_manifest)
    pull_manifest.add_argument("--split-path", required=True)
    pull_manifest.add_argument("--output", required=True)
    pull_manifest.add_argument("--profile")
    pull_manifest.add_argument("--upload", action="store_true")
    pull_manifest.add_argument("--dry-run", action="store_true")
    pull_manifest.set_defaults(func=cmd_s3_create_pull_manifest)

    pull_processed = s3_sub.add_parser("pull-processed", help="Download processed episodes from a static S3 pull manifest.")
    add_s3_config_arg(pull_processed)
    pull_processed.add_argument("--pull-manifest", required=True)
    pull_processed.add_argument("--dest-root", required=True)
    pull_processed.add_argument("--split", help="Only pull one split, e.g. train or holdout")
    pull_processed.add_argument("--dry-run", action="store_true")
    pull_processed.set_defaults(func=cmd_s3_pull_processed)
    return parser


def add_dataset_common(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--datasets", help="Comma-separated dataset slugs")
    parser.add_argument("--modalities", help="Comma-separated modalities: video,gaze,annotation,depth")
    parser.add_argument("--sequences", help="Comma-separated sequence ids")
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    return parser


def add_serve_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--canonical-root", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="Open the browser automatically")


def add_s3_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--s3-config", default="configs/s3.json", help="Path to user S3 config JSON")


def cmd_doctor(args: argparse.Namespace) -> int:
    report = {
        "python": sys.version.split()[0],
        "ffmpeg": shutil.which("ffmpeg"),
        "ffprobe": shutil.which("ffprobe"),
        "aws": shutil.which("aws"),
        "nfs_mount": "/nfs" if Path("/nfs").exists() else None,
        "parquet_available": parquet_available(),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["ffmpeg"]:
        print("warning: ffmpeg not found; video transcode/resample will require installing ffmpeg", file=sys.stderr)
    if not report["parquet_available"]:
        print("warning: pyarrow not found; tabular outputs will use explicit JSONL fallback files", file=sys.stderr)
    if not report["nfs_mount"]:
        print("warning: /nfs mount not found; default S3 tether commands expect s3://far-research-internal mounted at /nfs", file=sys.stderr)
    if not report["aws"]:
        print("warning: aws CLI not found; uploads to canonical S3 paths require aws s3 commands", file=sys.stderr)
    return 0


def cmd_datasets_plan(args: argparse.Namespace) -> int:
    catalog = load_catalog(args.repo_root)
    assets = selected_assets(catalog, args)
    rows = estimate_downloads(assets)
    emit(rows, as_json=args.json)
    return 0


def cmd_datasets_verify(args: argparse.Namespace) -> int:
    catalog = load_catalog(args.repo_root)
    checks = verify_links(selected_assets(catalog, args), sample_per_dataset=args.sample_per_dataset, timeout_s=args.timeout)
    rows = [check.__dict__ for check in checks]
    emit(rows, as_json=args.json)
    return 0 if all(row["ok"] for row in rows) else 1


def cmd_datasets_fetch(args: argparse.Namespace) -> int:
    catalog = load_catalog(args.repo_root)
    assets = selected_assets(catalog, args)
    if args.manifest_out:
        write_download_manifest(assets, args.manifest_out)
    rows = fetch_assets(assets, args.raw_root, dry_run=args.dry_run)
    emit(rows, as_json=args.json)
    return 0 if all("error" not in row for row in rows) else 1


def cmd_rectify(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, args.overrides)
    rows = rectify_dataset(
        args.raw_root,
        args.canonical_root,
        config=cfg,
        dataset=args.dataset,
        episodes=parse_set(args.episodes),
    )
    emit(rows, as_json=True)
    return 0


def cmd_validate_alignment(args: argparse.Namespace) -> int:
    report = validate_canonical_root(args.canonical_root, raw_root=args.raw_root)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


def cmd_split_create(args: argparse.Namespace) -> int:
    request = SplitRequest(
        name=args.name,
        ratios=parse_ratios(args.ratios),
        seed=args.seed,
        mode=args.mode,
        include_datasets=parse_set(args.include_datasets),
        include_modalities=parse_set(args.include_modalities),
        group_by=args.group_by,
        stratify_by=args.stratify_by,
    )
    print(json.dumps(create_split(args.canonical_root, request), indent=2, sort_keys=True))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    serve(args.canonical_root, host=args.host, port=args.port, open_browser=args.open)
    return 0


def cmd_s3_layout(args: argparse.Namespace) -> int:
    cfg = load_s3_config(args.s3_config)
    print(
        json.dumps(
            {
                "layout_version": 1,
                "access_mode": cfg.access_mode,
                "upload_mode": cfg.upload_mode,
                "base": cfg.base_uri(),
                "mount_root": cfg.mount_root,
                "mounted_base_for_access": str(mount_path_for_uri(cfg, cfg.base_uri())) if cfg.access_mode == "file_mount" else None,
                "unprocessed": cfg.unprocessed_uri("{dataset}", "{partition}", "{asset_key}"),
                "unprocessed_access_path": str(mount_path_for_uri(cfg, cfg.unprocessed_uri("{dataset}", "{partition}", "{asset_key}")))
                if cfg.access_mode == "file_mount"
                else None,
                "processed_episode": cfg.processed_uri("{profile}", "{dataset}", "{episode_id}"),
                "processed_episode_access_path": str(mount_path_for_uri(cfg, cfg.processed_uri("{profile}", "{dataset}", "{episode_id}")))
                if cfg.access_mode == "file_mount"
                else None,
                "processed_manifest": cfg.processed_manifest_uri("{profile}"),
                "download_manifest": cfg.download_manifest_uri("{name}"),
                "split_pull_manifest": cfg.split_uri("{name}"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def cmd_s3_backup_raw(args: argparse.Namespace) -> int:
    cfg = load_s3_config(args.s3_config)
    report = serial_download_backup(
        args.repo_root,
        cfg,
        args.raw_root,
        datasets=parse_set(args.datasets),
        modalities=parse_set(args.modalities),
        sequences=parse_set(args.sequences),
        dry_run=args.dry_run,
        manifest_name=args.manifest_name,
        max_download_bytes=args.max_download_bytes,
        reserve_bytes=args.reserve_bytes,
        storage_fraction=args.storage_fraction,
        include_unknown_size=args.include_unknown_size,
        clean_after_upload=not args.keep_cache,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if all("error" not in operation.get("fetch", {}) for operation in report["operations"]) else 1


def cmd_s3_process_serial(args: argparse.Namespace) -> int:
    cfg = load_s3_config(args.s3_config)
    report = serial_process_backup(
        cfg,
        parse_partitions(args.partitions),
        config=load_config(args.config) if args.config else None,
        local_cache_root=args.local_cache_root,
        dry_run=args.dry_run,
        clean=not args.keep_cache,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def cmd_s3_create_pull_manifest(args: argparse.Namespace) -> int:
    cfg = load_s3_config(args.s3_config)
    report = create_s3_pull_manifest(
        cfg,
        args.split_path,
        args.output,
        profile=args.profile,
        upload=args.upload,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def cmd_s3_pull_processed(args: argparse.Namespace) -> int:
    cfg = load_s3_config(args.s3_config)
    report = pull_processed_from_manifest(
        cfg,
        args.pull_manifest,
        args.dest_root,
        split=args.split,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def selected_assets(catalog, args: argparse.Namespace):
    return plan_downloads(
        catalog,
        datasets=parse_set(args.datasets),
        modalities=parse_set(args.modalities),
        sequences=parse_set(args.sequences),
    )


def parse_set(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def parse_ratios(value: str) -> dict[str, float]:
    result = {}
    for part in value.split(","):
        if not part.strip():
            continue
        key, raw = part.split("=", 1)
        result[key.strip()] = float(raw)
    return result


def emit(rows, as_json: bool = False) -> None:
    if as_json:
        print(json.dumps(rows, indent=2, sort_keys=True))
        return
    if not rows:
        print("No rows")
        return
    columns = list(rows[0])
    widths = {column: max(len(column), *(len(str(row.get(column, ""))) for row in rows)) for column in columns}
    print("  ".join(column.ljust(widths[column]) for column in columns))
    print("  ".join("-" * widths[column] for column in columns))
    for row in rows:
        print("  ".join(str(row.get(column, "")).ljust(widths[column]) for column in columns))


if __name__ == "__main__":
    raise SystemExit(main())
