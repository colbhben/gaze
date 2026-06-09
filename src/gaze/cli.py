from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

from . import __version__
from .config import load_config
from .manifest import copy_manifest, inspect_raw_root, wait_for_downloads_and_write_manifest, write_manifest
from .rectify import rectify_dataset
from .s3 import (
    create_s3_pull_manifest,
    load_s3_config,
    mount_path_for_uri,
    parse_partitions,
    pull_processed_from_manifest,
    serial_process_backup,
)
from .server import serve
from .splits import SplitRequest, create_split
from .table import parquet_available
from .validate import validate_canonical_root


HELP_NOTES = """Common value formats:
  Comma-separated filters have no spaces: --datasets aea,hot3d --modalities video,gaze
  Repeated overrides use dotted keys: --set target_hz=5 --set video.width=392
  Paths may be relative to the current directory or absolute.

Subcommand help:
  Use -h after any command group or command to see its focused options.
  Examples: gaze datasets -h, gaze rectify -h, gaze s3 process-serial -h
"""

DATASET_FILTER_HELP = """Dataset filters:
  --datasets limits work to dataset slugs such as aea, hot3d, nymeria,
    adt, holoassist, egtea, or ego-exo4d. Omit it to include every cataloged dataset.
  --modalities limits assets by normalized modality: video, gaze, annotation,
    depth, pose, or other. Omit it to include all modalities.
  --sequences limits assets to exact sequence ids from provider manifests.
  --json switches table output to JSON for scripts.
"""

RECTIFY_OVERRIDE_HELP = """Override examples:
  --set target_hz=5
  --set profile_name=qwen3-vl-gaze-5hz-392px
  --set video.fps=5 --set video.width=392 --set video.height=392
  --set video.resize_mode=pad --set depth.enabled=false

Supported config groups include profile_name, target_hz, video.*, gaze.*,
annotation.*, depth.*, and validation.*. See src/gaze/config.py for defaults.
"""

S3_CONFIG_HELP = """S3 config:
  The JSON file describes the bucket URI, mount root, upload/access modes,
  local cache budget, profile name, and optional AWS profile/region settings.
  Copy configs/s3.example.json to configs/s3.json before customizing it.
"""

class GazeArgumentParser(argparse.ArgumentParser):
    """ArgumentParser with command-local examples and friendlier errors."""

    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(
            2,
            f"{self.prog}: error: {message}\n"
            f"Run '{self.prog} -h' to see available options, value formats, and examples. "
            "If the error followed a subcommand, run that subcommand with -h.\n",
        )


def name_positionals(parser: argparse.ArgumentParser, title: str, description: str | None = None) -> None:
    parser._positionals.title = title
    parser._positionals.description = description


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        print("\nChoose one command above, for example: gaze doctor", file=sys.stderr)
        return 2
    try:
        return args.func(args)
    except Exception as exc:
        print(
            f"error: {exc}\n"
            "Run the same command with -h for option details and examples.",
            file=sys.stderr,
        )
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = GazeArgumentParser(
        prog="gaze",
        description=(
            "Inspect, rectify, validate, split, and view heterogeneous gaze datasets. "
            "Use a subcommand with -h to see the options available for that workflow."
        ),
        epilog=HELP_NOTES + "\nExamples:\n  gaze doctor\n  gaze datasets inspect-manifest --raw-root ./raw --manifest-out ./raw_manifest.json\n  gaze rectify --raw-root ./raw --canonical-root ./canonical --dataset toy\n  gaze view --canonical-root ./canonical",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    name_positionals(parser, "commands", "Choose one command, then add that command's options.")
    parser.add_argument("--version", action="version", version=f"gaze {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>", parser_class=GazeArgumentParser)

    doctor = sub.add_parser(
        "doctor",
        help="Check local runtime dependencies.",
        description="Print a JSON report describing local tools and optional runtime features used by gaze.",
        epilog="Example:\n  gaze doctor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    doctor.set_defaults(func=cmd_doctor)

    datasets = sub.add_parser(
        "datasets",
        help="Inspect downloaded raw datasets and write raw-format manifests.",
        description="Traverse a downloaded raw dataset root and write a raw/canonical manifest, or watch a raw root until downloads land before writing one.",
        epilog=DATASET_FILTER_HELP
        + "\nExamples:\n  gaze datasets inspect-manifest --raw-root .gaze-cache/raw --manifest-out .gaze-cache/raw_manifest.json\n  gaze datasets watch-manifest --raw-root .gaze-cache/raw --manifest-out .gaze-cache/raw_manifest.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    name_positionals(datasets, "dataset commands", "Choose one dataset command.")
    datasets_sub = datasets.add_subparsers(dest="datasets_command", metavar="<datasets-command>", required=True, parser_class=GazeArgumentParser)

    inspect_manifest = add_dataset_common(
        datasets_sub.add_parser(
            "inspect-manifest",
            help="Inspect downloaded raw datasets and write a raw-format manifest.",
            description="Traverse a raw dataset root, infer observed file structures and data formats, and write a JSON manifest that also records the canonical rectified layout.",
            epilog=DATASET_FILTER_HELP
            + "\nExample:\n  gaze datasets inspect-manifest --raw-root .gaze-cache/raw --manifest-out .gaze-cache/raw_manifest.json",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
    )
    inspect_manifest.add_argument("--raw-root", metavar="PATH", required=True, help="Raw dataset root to inspect. Required.")
    inspect_manifest.add_argument("--manifest-out", metavar="PATH", required=True, help="JSON manifest to write. Required.")
    inspect_manifest.add_argument("--copy-out", metavar="PATH", help="Optional second local path to copy the manifest to after writing.")
    inspect_manifest.set_defaults(func=cmd_datasets_inspect_manifest)

    watch_manifest = add_dataset_common(
        datasets_sub.add_parser(
            "watch-manifest",
            help="Wait for downloads to finish, then write the raw-format manifest.",
            description="Poll the raw root until every catalog-selected asset is present and stable, traverse the raw datasets once, and write the inferred raw/canonical manifest.",
            epilog=DATASET_FILTER_HELP
            + "\nExample:\n  gaze datasets watch-manifest --raw-root .gaze-cache/raw --manifest-out .gaze-cache/raw_manifest.json --poll-seconds 60",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
    )
    watch_manifest.add_argument("--raw-root", metavar="PATH", required=True, help="Raw dataset root to watch. Required.")
    watch_manifest.add_argument("--manifest-out", metavar="PATH", required=True, help="JSON manifest to write once downloads are complete. Required.")
    watch_manifest.add_argument("--copy-out", metavar="PATH", help="Optional second local path to copy the completed manifest to after writing.")
    watch_manifest.add_argument("--poll-seconds", metavar="SECONDS", type=float, default=60.0, help="Seconds between completion checks; default: 60.")
    watch_manifest.add_argument("--stable-checks", metavar="N", type=int, default=2, help="Consecutive complete checks required before inspection; default: 2.")
    watch_manifest.add_argument("--timeout-seconds", metavar="SECONDS", type=float, help="Optional maximum watch time before failing.")
    watch_manifest.set_defaults(func=cmd_datasets_watch_manifest)

    rectify = sub.add_parser(
        "rectify",
        help="Rectify raw fixture/dataset roots into canonical episodes.",
        description="Read raw episodes, resample them to the configured timeline, and write the canonical episode layout.",
        epilog=RECTIFY_OVERRIDE_HELP
        + "\nExamples:\n  gaze rectify --raw-root ./raw --canonical-root ./canonical\n  gaze rectify --raw-root ./raw --canonical-root ./canonical --dataset toy --episodes ep1,ep2 --set target_hz=5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    rectify.add_argument("--raw-root", metavar="PATH", required=True, help="Directory containing raw dataset folders or fixture episodes. Required.")
    rectify.add_argument("--canonical-root", metavar="PATH", required=True, help="Output directory for canonical episodes, manifests, and tables. Required.")
    rectify.add_argument("--dataset", metavar="SLUG", help="Only rectify one dataset slug under the raw root, for example toy or aea. Omit for all datasets.")
    rectify.add_argument("--episodes", metavar="ID[,ID...]", help="Comma-separated episode ids to rectify within the selected dataset(s), for example ep1,ep2.")
    rectify.add_argument("--config", metavar="PATH", help="Path to a rectification JSON config. Omit to use built-in defaults.")
    rectify.add_argument("--raw-manifest", metavar="PATH", help="Raw-format manifest from datasets inspect-manifest/watch-manifest. Enables rectification from observed dataset structure when episode.json files are absent.")
    rectify.add_argument("--set", metavar="KEY=VALUE", action="append", default=[], dest="overrides", help="Override one config value. Repeat for multiple overrides, for example --set target_hz=5 --set video.resize_mode=pad.")
    rectify.set_defaults(func=cmd_rectify)

    validate = sub.add_parser(
        "validate",
        help="Validation commands.",
        description="Run checks that compare canonical outputs against expected structure and raw-source alignment.",
        epilog="Example:\n  gaze validate alignment --canonical-root ./canonical --raw-root ./raw",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    name_positionals(validate, "validation commands", "Choose one validation command.")
    validate_sub = validate.add_subparsers(dest="validate_command", metavar="<validate-command>", required=True, parser_class=GazeArgumentParser)
    alignment = validate_sub.add_parser(
        "alignment",
        help="Validate raw-to-canonical alignment.",
        description="Validate canonical episode timelines, modality tables, video frame counts, and optional raw-source preservation.",
        epilog="Examples:\n  gaze validate alignment --canonical-root ./canonical\n  gaze validate alignment --canonical-root ./canonical --raw-root ./raw",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    alignment.add_argument("--canonical-root", metavar="PATH", required=True, help="Canonical root produced by gaze rectify. Required.")
    alignment.add_argument("--raw-root", metavar="PATH", help="Optional raw root used to compare canonical outputs against raw inputs.")
    alignment.set_defaults(func=cmd_validate_alignment)

    split = sub.add_parser(
        "split",
        help="Split commands.",
        description="Create deterministic train/holdout manifests from canonical episodes.",
        epilog="Example:\n  gaze split create --canonical-root ./canonical --name default --ratios train=0.8,holdout=0.2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    name_positionals(split, "split commands", "Choose one split command.")
    split_sub = split.add_subparsers(dest="split_command", metavar="<split-command>", required=True, parser_class=GazeArgumentParser)
    create = split_sub.add_parser(
        "create",
        help="Create seeded train/holdout manifests.",
        description="Create a named split manifest under the canonical root using deterministic seeded assignment.",
        epilog=(
            "Split arguments:\n"
            "  --mode heterogeneous allows datasets/modalities to mix across split buckets.\n"
            "  --mode homogeneous keeps grouped units together based on --group-by.\n"
            "  --ratios is a comma-separated map whose values should sum to 1.0.\n\n"
            "Examples:\n"
            "  gaze split create --canonical-root ./canonical --name demo\n"
            "  gaze split create --canonical-root ./canonical --include-datasets aea,hot3d --include-modalities video,gaze --seed 42"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    create.add_argument("--canonical-root", metavar="PATH", required=True, help="Canonical root containing rectified episodes. Required.")
    create.add_argument("--name", metavar="NAME", default="default", help="Split name and output filename stem under <canonical-root>/splits; default: default.")
    create.add_argument("--ratios", metavar="NAME=FLOAT[,NAME=FLOAT...]", default="train=0.8,holdout=0.2", help="Comma-separated split ratios, for example train=0.8,holdout=0.2; default: train=0.8,holdout=0.2.")
    create.add_argument("--seed", metavar="INT", type=int, default=0, help="Integer random seed for deterministic split assignment; default: 0.")
    create.add_argument("--mode", choices=["heterogeneous", "homogeneous"], default="heterogeneous", help="Split strategy. Choices: heterogeneous, homogeneous. Default: heterogeneous.")
    create.add_argument("--include-datasets", metavar="SLUG[,SLUG...]", help="Only include these dataset slugs in the split, for example aea,hot3d.")
    create.add_argument("--include-modalities", metavar="MOD[,MOD...]", help="Only include episodes with these modalities, for example video,gaze.")
    create.add_argument("--group-by", metavar="FIELD", default="dataset", help="Episode metadata field used for homogeneous grouping; default: dataset.")
    create.add_argument("--stratify-by", metavar="FIELD", help="Optional episode metadata field used to balance split assignment.")
    create.set_defaults(func=cmd_split_create)

    serve_parser = sub.add_parser(
        "serve",
        help="Serve the local API and browser viewer.",
        description="Start the local HTTP API and static browser viewer for a canonical root.",
        epilog="Example:\n  gaze serve --canonical-root ./canonical --host 127.0.0.1 --port 8765",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_serve_args(serve_parser)
    serve_parser.set_defaults(func=cmd_serve)

    view = sub.add_parser(
        "view",
        help="Launch the local browser viewer.",
        description="Start the local viewer for a canonical root and open it in the default browser.",
        epilog="Example:\n  gaze view --canonical-root ./canonical",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_serve_args(view)
    view.set_defaults(open=True, func=cmd_serve)

    s3 = sub.add_parser(
        "s3",
        help="S3-backed serial pipeline commands.",
        description="Run serial S3 workflows that process raw partitions, create pull manifests, and pull processed data.",
        epilog=S3_CONFIG_HELP
        + "\nExamples:\n  gaze s3 layout --s3-config configs/s3.json\n  gaze s3 process-serial --s3-config configs/s3.json --partitions toy:ep1,toy:ep2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    name_positionals(s3, "S3 commands", "Choose one S3 command.")
    s3_sub = s3.add_subparsers(dest="s3_command", metavar="<s3-command>", required=True, parser_class=GazeArgumentParser)
    layout = s3_sub.add_parser(
        "layout",
        help="Show the configured static S3 layout.",
        description="Print the S3 URI layout and, when configured, the local mount paths used for read access.",
        epilog=S3_CONFIG_HELP + "\nExample:\n  gaze s3 layout --s3-config configs/s3.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_s3_config_arg(layout)
    layout.set_defaults(func=cmd_s3_layout)

    process_serial = s3_sub.add_parser(
        "process-serial",
        help="Serially pull raw partitions from S3, rectify, and upload processed episodes.",
        description="Copy configured raw partitions from S3/mount access into local cache, rectify them, upload processed episodes, and update the processed manifest.",
        epilog=S3_CONFIG_HELP
        + "\nExamples:\n  gaze s3 process-serial --s3-config configs/s3.json --partitions toy:ep1,toy:ep2\n  gaze s3 process-serial --partitions aea:loc1_script1_seq1_rec1 --config ./rectify.json --keep-cache",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_s3_config_arg(process_serial)
    process_serial.add_argument("--partitions", metavar="DATASET:PARTITION[,DATASET:PARTITION...]", required=True, help="Comma-separated raw partitions to process, for example toy:ep1,toy:ep2. Required.")
    process_serial.add_argument("--local-cache-root", metavar="PATH", help="Override the local cache root from the S3 config for this run.")
    process_serial.add_argument("--config", metavar="PATH", help="Path to a rectification JSON config. Omit to use built-in defaults.")
    process_serial.add_argument("--dry-run", action="store_true", help="Show the processing/upload plan without copying, rectifying, uploading, or deleting files.")
    process_serial.add_argument("--keep-cache", action="store_true", help="Keep local per-partition cache directories after successful upload.")
    process_serial.set_defaults(func=cmd_s3_process_serial)

    pull_manifest = s3_sub.add_parser(
        "create-pull-manifest",
        help="Create a static S3 pull manifest from a train/holdout split.",
        description="Convert a local split manifest into a static manifest with S3/mount source paths for processed episode pulling.",
        epilog=S3_CONFIG_HELP
        + "\nExample:\n  gaze s3 create-pull-manifest --s3-config configs/s3.json --split-path ./canonical/splits/demo.json --output .gaze-cache/splits/demo.s3.json --upload",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_s3_config_arg(pull_manifest)
    pull_manifest.add_argument("--split-path", metavar="PATH", required=True, help="Local split JSON created by gaze split create. Required.")
    pull_manifest.add_argument("--output", metavar="PATH", required=True, help="Local path where the S3 pull manifest JSON should be written. Required.")
    pull_manifest.add_argument("--profile", metavar="NAME", help="Processed profile name to reference; defaults to the profile configured in the S3 config.")
    pull_manifest.add_argument("--upload", action="store_true", help="Upload the generated pull manifest to the configured S3 splits prefix.")
    pull_manifest.add_argument("--dry-run", action="store_true", help="Print the planned manifest/upload report without writing or uploading.")
    pull_manifest.set_defaults(func=cmd_s3_create_pull_manifest)

    pull_processed = s3_sub.add_parser(
        "pull-processed",
        help="Download processed episodes from a static S3 pull manifest.",
        description="Read a static pull manifest and copy selected processed episodes into a local destination root.",
        epilog="Examples:\n  gaze s3 pull-processed --s3-config configs/s3.json --pull-manifest .gaze-cache/splits/demo.s3.json --dest-root ./canonical_train\n  gaze s3 pull-processed --pull-manifest ./demo.s3.json --split train --dest-root ./canonical_train --dry-run",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_s3_config_arg(pull_processed)
    pull_processed.add_argument("--pull-manifest", metavar="PATH", required=True, help="Static S3 pull manifest created by gaze s3 create-pull-manifest. Required.")
    pull_processed.add_argument("--dest-root", metavar="PATH", required=True, help="Local destination root for copied processed episodes. Required.")
    pull_processed.add_argument("--split", metavar="NAME", help="Only pull one split bucket, for example train or holdout. Omit to pull all buckets.")
    pull_processed.add_argument("--dry-run", action="store_true", help="Show the planned copies without writing files.")
    pull_processed.set_defaults(func=cmd_s3_pull_processed)
    return parser


def add_dataset_common(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--repo-root", metavar="PATH", default=".", help="Repository root containing DATASETS.md and download_links; default: current directory.")
    parser.add_argument("--datasets", metavar="SLUG[,SLUG...]", help="Comma-separated dataset slugs to include, for example aea,hot3d. Omit to include all datasets.")
    parser.add_argument("--modalities", metavar="MOD[,MOD...]", help="Comma-separated modalities to include: video,gaze,annotation,depth,pose,other. Omit to include all modalities.")
    parser.add_argument("--sequences", metavar="ID[,ID...]", help="Comma-separated provider sequence ids to include exactly. Omit to include all sequences.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of the default human-readable table.")
    return parser


def add_serve_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--canonical-root", metavar="PATH", required=True, help="Canonical root containing rectified episodes and manifest files. Required.")
    parser.add_argument("--host", metavar="HOST", default="127.0.0.1", help="Interface to bind the local HTTP server; default: 127.0.0.1.")
    parser.add_argument("--port", metavar="PORT", type=int, default=8765, help="TCP port for the local HTTP server; default: 8765.")
    parser.add_argument("--open", action="store_true", help="Open the viewer URL in the default browser after the server starts.")


def add_s3_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--s3-config", metavar="PATH", default="configs/s3.json", help="Path to user S3 config JSON; default: configs/s3.json.")


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


def cmd_datasets_inspect_manifest(args: argparse.Namespace) -> int:
    manifest = inspect_raw_root(
        args.raw_root,
        repo_root=args.repo_root,
        datasets=parse_set(args.datasets),
        modalities=parse_set(args.modalities),
        sequences=parse_set(args.sequences),
    )
    written = write_manifest(manifest, args.manifest_out)
    if args.copy_out:
        copy_manifest(written, args.copy_out)
    print(json.dumps({"manifest": args.manifest_out, "copy": args.copy_out, "datasets": sorted(manifest["datasets"])}, indent=2, sort_keys=True))
    return 0


def cmd_datasets_watch_manifest(args: argparse.Namespace) -> int:
    manifest = wait_for_downloads_and_write_manifest(
        args.raw_root,
        args.manifest_out,
        copy_out=args.copy_out,
        repo_root=args.repo_root,
        datasets=parse_set(args.datasets),
        modalities=parse_set(args.modalities),
        sequences=parse_set(args.sequences),
        poll_seconds=args.poll_seconds,
        stable_checks=args.stable_checks,
        timeout_seconds=args.timeout_seconds,
    )
    print(
        json.dumps(
            {
                "manifest": args.manifest_out,
                "copy": args.copy_out,
                "datasets": sorted(manifest["datasets"]),
                "watcher": manifest.get("watcher", {}),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def cmd_rectify(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, args.overrides)
    rows = rectify_dataset(
        args.raw_root,
        args.canonical_root,
        config=cfg,
        dataset=args.dataset,
        episodes=parse_set(args.episodes),
        raw_manifest=args.raw_manifest,
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
                "split_pull_manifest": cfg.split_uri("{name}"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


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
