from __future__ import annotations

import argparse
import json
import os
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

    add_curate_commands(sub)
    return parser


CURATE_HELP = """Recipe-driven curation:
  Recipes live in recipes/<slug>.json and capture how to parse/extract each
  dataset (episode enumeration, video/gaze/annotation selection, native gaze
  format + projection, multi-channel annotations, epoch reconciliation).
  Source data is read-only on the remote NFS host (pulled via ssh/scp); the
  local machine is the processing host (ffprobe, projectaria_tools, ffmpeg).
"""


def add_curate_commands(sub: argparse._SubParsersAction) -> None:
    curate = sub.add_parser(
        "curate",
        help="Recipe-driven extraction, overlay, and smoke-manifest commands.",
        description="Extract one episode per recipe, render gaze+annotation overlays, assemble a viewer-ready smoke manifest, and upload it.",
        epilog=CURATE_HELP
        + "\nExamples:\n  gaze curate extract --dataset nymeria --episode 20230607_s1_barbara_wheeler_act1_nkg6zo\n  gaze curate smoke --upload",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    name_positionals(curate, "curate commands", "Choose one curate command.")
    curate_sub = curate.add_subparsers(dest="curate_command", metavar="<curate-command>", required=True, parser_class=GazeArgumentParser)

    extract = curate_sub.add_parser(
        "extract",
        help="Extract one episode's video metadata, gaze, and annotation channels.",
        description="Resolve a recipe's episode files from the read-only source, parse gaze + annotations, ffprobe video, and write the episode bundle JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    extract.add_argument("--dataset", metavar="SLUG", required=True, help="Dataset slug (recipe name). Required.")
    extract.add_argument("--episode", metavar="ID", help="Episode id. Omit to use recipes/_sample_episodes.json.")
    extract.add_argument("--out-dir", metavar="PATH", default="/tmp/gaze_extract", help="Output dir for <slug>.json + <slug>_full.json; default: /tmp/gaze_extract.")
    extract.add_argument("--ssh-host", metavar="HOST", default="sumedhso-L40S", help="Remote data host; default: sumedhso-L40S.")
    extract.add_argument("--local-root", metavar="PATH", help="Read source from a local mount instead of ssh (e.g. a mounted /nfs).")
    extract.set_defaults(func=cmd_curate_extract)

    overlay = curate_sub.add_parser(
        "overlay",
        help="Render a gaze+annotation overlay clip for one episode.",
        description="Project gaze per frame (per the recipe), draw gaze + active annotation captions, and encode a short overlay mp4.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    overlay.add_argument("--dataset", metavar="SLUG", required=True, help="Dataset slug. Required.")
    overlay.add_argument("--episode", metavar="ID", help="Episode id. Omit to use the sample episode.")
    overlay.add_argument("--out", metavar="PATH", help="Output mp4 path; default: /tmp/gaze_overlays/<slug>.mp4.")
    overlay.add_argument("--max-seconds", metavar="N", type=float, default=20.0, help="Clip length cap; default: 20.")
    overlay.add_argument("--ssh-host", metavar="HOST", default="sumedhso-L40S", help="Remote data host; default: sumedhso-L40S.")
    overlay.add_argument("--local-root", metavar="PATH", help="Read source from a local mount instead of ssh.")
    overlay.set_defaults(func=cmd_curate_overlay)

    build_smoke = curate_sub.add_parser(
        "build-smoke",
        help="Assemble a viewer-ready smoke manifest from extracted episodes + overlays.",
        description="Build a canonical-style root (manifest + episodes/<dataset>/<id>/ with overlay.mp4, gaze, annotations) the gaze viewer can serve.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    build_smoke.add_argument("--extract-dir", metavar="PATH", default="/tmp/gaze_extract", help="Dir of extracted bundles; default: /tmp/gaze_extract.")
    build_smoke.add_argument("--overlays-dir", metavar="PATH", default="/tmp/gaze_overlays", help="Dir of overlay mp4s; default: /tmp/gaze_overlays.")
    build_smoke.add_argument("--out-root", metavar="PATH", default="/tmp/gaze_smoke_manifest", help="Output root; default: /tmp/gaze_smoke_manifest.")
    build_smoke.add_argument("--datasets", metavar="SLUG[,SLUG...]", help="Only include these dataset slugs.")
    build_smoke.set_defaults(func=cmd_curate_build_smoke)

    upload_smoke = curate_sub.add_parser(
        "upload-smoke",
        help="Upload a smoke manifest to S3 via the remote host.",
        description="scp the smoke root to the remote host and `aws s3 sync` it to the smoke_manifest prefix (local AWS access is read-only here).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    upload_smoke.add_argument("--root", metavar="PATH", default="/tmp/gaze_smoke_manifest", help="Local smoke root to upload; default: /tmp/gaze_smoke_manifest.")
    upload_smoke.add_argument("--ssh-host", metavar="HOST", default="sumedhso-L40S", help="Remote host with S3 write access; default: sumedhso-L40S.")
    upload_smoke.add_argument("--s3-uri", metavar="URI", default="s3://far-research-internal/colbhben/gaze/unprocessed/smoke_manifest", help="Destination S3 prefix.")
    upload_smoke.add_argument("--dry-run", action="store_true", help="Show the upload plan without copying.")
    upload_smoke.set_defaults(func=cmd_curate_upload_smoke)

    smoke = curate_sub.add_parser(
        "smoke",
        help="End-to-end: build the smoke manifest from existing artifacts (and optionally upload).",
        description="Assemble the smoke manifest from already-extracted bundles + overlays, print the report, and optionally upload to S3.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    smoke.add_argument("--extract-dir", metavar="PATH", default="/tmp/gaze_extract")
    smoke.add_argument("--overlays-dir", metavar="PATH", default="/tmp/gaze_overlays")
    smoke.add_argument("--out-root", metavar="PATH", default="/tmp/gaze_smoke_manifest")
    smoke.add_argument("--upload", action="store_true", help="Upload the assembled manifest to S3 after building.")
    smoke.add_argument("--ssh-host", metavar="HOST", default="sumedhso-L40S")
    smoke.add_argument("--s3-uri", metavar="URI", default="s3://far-research-internal/colbhben/gaze/unprocessed/smoke_manifest")
    smoke.add_argument("--dry-run", action="store_true", help="With --upload, show the upload plan without copying.")
    smoke.set_defaults(func=cmd_curate_smoke)

    build_training = curate_sub.add_parser(
        "build-training-manifest",
        help="Build a Molmo2 (or Qwen) gaze training manifest from annotation-bounded clips.",
        description=(
            "Default output-format=molmo2: chop each episode into annotation-bounded clip "
            "SEGMENTS (capped at --max-clip-duration-s), materialize each as a resolution^2 @ fps "
            "mp4, and emit Molmo2VideoPoint rows (message_list + per-frame points + timestamps) "
            "with gaze_hz == video_fps: EXACTLY ONE pixel gaze point per video frame (1:1, variable "
            "length, no cap/pad). --max-frames>0 optionally caps clip length. dataset_filters "
            "(cull / exclude globs / gaze-gap) are honored. --min-duration-s coalesces too-short "
            "clips. output-format=qwen keeps the legacy fixed sliding-window single-point profile."
        ),
        epilog=(
            "Defaults: 2 fps (Molmo2 video-pointing path is natively 6 fps; any int <=10 is loader-valid),\n"
            "378x378, max clip 20s, one gaze point per video frame.\n"
            "Examples:\n"
            "  gaze curate build-training-manifest\n"
            "  gaze curate build-training-manifest --datasets nymeria --interesting-map nymeria=/tmp/nym_map.json\n"
            "  gaze curate build-training-manifest --output-format qwen --fps 5 --num-frames 16"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    build_training.add_argument("--output-format", choices=["molmo2", "qwen"], default="molmo2", help="Manifest format; default: molmo2.")
    build_training.add_argument("--profile", metavar="NAME", default="qwen3-vl-gaze-5hz-392px", help="(qwen) canonical profile name.")
    build_training.add_argument("--fps", metavar="HZ", type=float, default=2.0, help="Canonical sampling fps; ALL datasets resample down to this. Default: 2. NOTE: Molmo2's video-pointing/track path is natively 6 fps (TrackingDataset.VIDEO_FPS=6); any integer <=10 is loader-valid (no 0.5s grid constraint). Consider --fps 6 to match Molmo2's in-distribution pointing rate.")
    build_training.add_argument("--max-frames", metavar="N", type=int, default=0, help="(molmo2) OPTIONAL cap: 0/unset = one gaze point per video frame (gaze_hz==fps, no cap). N>0 caps clips to N/fps seconds. Default: 0 (unlimited).")
    build_training.add_argument("--max-clip-duration-s", metavar="S", type=float, default=20.0, help="(molmo2) max clip/segment length in seconds; long takes split at annotation bounds. Default: 20.")
    build_training.add_argument("--merge-gap-s", metavar="S", type=float, default=1.0, help="(molmo2) merge annotation spans separated by <= this gap. Default: 1.")
    build_training.add_argument("--drop-shorter-than-s", metavar="S", type=float, default=1.0, help="(molmo2) drop chopped segments shorter than this. Default: 1.")
    build_training.add_argument("--min-duration-s", metavar="S", type=float, default=0.0, help="(molmo2) coalesce a clip shorter than this with the next clip(s) (text -> numbered list), up to --max-clip-duration-s. 0/unset = disabled. Default: 0.")
    build_training.add_argument("--prompt", metavar="TEXT", default="Point to where the camera wearer is looking.", help="(molmo2) gaze prompt text.")
    build_training.add_argument("--workers", metavar="N", type=int, help="(molmo2) parallel episodes; default: CPU count.")
    build_training.add_argument("--num-frames", metavar="N", type=int, default=16, help="(qwen) frames per clip; default: 16.")
    build_training.add_argument("--stride", metavar="N", type=int, default=8, help="(qwen) anchor hop in frames; default: 8.")
    build_training.add_argument("--resolution", metavar="PX", type=int, default=378, help="Square padded video side in pixels; default: 378 (Molmo2).")
    build_training.add_argument("--temporality", choices=["causal", "centered", "future"], default="causal", help="(qwen) clip window vs anchor; default: causal.")
    build_training.add_argument("--window-seconds", metavar="N", type=float, default=30.0, help="(qwen) max source seconds to trim+resample per episode; default: 30.")
    build_training.add_argument("--out-root", metavar="PATH", default="/tmp/gaze_training_manifest", help="Output root; default: /tmp/gaze_training_manifest.")
    build_training.add_argument("--datasets", metavar="SLUG[,SLUG...]", help="Only build these dataset slugs. Omit for all sample datasets.")
    build_training.add_argument("--episode", metavar="ID", help="Override the episode id (use with a single --datasets slug).")
    build_training.add_argument("--episodes-file", metavar="PATH", help="JSON {datasets:{slug:[episode_id,...]}} for multiple episodes per dataset.")
    build_training.add_argument("--interesting-map", metavar="SLUG=PATH", action="append", default=[], help="Per-dataset interesting-region filter map (repeatable), e.g. nymeria=/tmp/map.json.")
    build_training.add_argument("--sample", metavar="PATH", help="Alternate sample-episodes JSON.")
    build_training.add_argument("--ssh-host", metavar="HOST", default="sumedhso-L40S", help="Remote data host; default: sumedhso-L40S.")
    build_training.add_argument("--local-root", metavar="PATH", help="Read source from a local mount instead of ssh.")
    build_training.add_argument("--no-reuse-bundle", action="store_true", help="Re-extract from source instead of reusing /tmp/gaze_extract/<slug>_full.json.")
    build_training.add_argument("--reuse-clips", action="store_true", help="Reuse already-encoded segment mp4s in --out-root (skip ffmpeg re-encode + skip the big source pull when all clips present). For recovering a manifest after a crash where clips are on disk but manifest.jsonl was never written.")
    build_training.set_defaults(func=cmd_curate_build_training)

    export_anno = curate_sub.add_parser(
        "export-annotations",
        help="Export a dataset's annotation spans (text+times) for interesting-region classification.",
        description=(
            "Flatten each episode's annotation channels into a JSONL of spans for LLM "
            "classification (e.g. nymeria interesting-region filtering). One record per episode: "
            "{take_id, spans:[{i,start_s,end_s,channel,text}]}. Feed to Claude (subagents for smoke; "
            "scripts/classify_nymeria_regions.py on Bedrock for the full dataset) to produce the "
            "filter map consumed by build-training-manifest --interesting-map."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    export_anno.add_argument("--dataset", metavar="SLUG", default="nymeria", help="Dataset slug; default: nymeria.")
    export_anno.add_argument("--episodes-file", metavar="PATH", help="JSON {datasets:{slug:[ep,...]}}; omit to use the sample episode.")
    export_anno.add_argument("--episode", metavar="ID", help="Single episode id (alternative to --episodes-file).")
    export_anno.add_argument("--out", metavar="PATH", default="/tmp/gaze_anno_export.jsonl", help="Output JSONL; default: /tmp/gaze_anno_export.jsonl.")
    export_anno.add_argument("--ssh-host", metavar="HOST", default="sumedhso-L40S", help="Remote data host; default: sumedhso-L40S.")
    export_anno.add_argument("--local-root", metavar="PATH", help="Read source from a local mount instead of ssh.")
    export_anno.set_defaults(func=cmd_curate_export_annotations)

    viewer = curate_sub.add_parser(
        "viewer-layout",
        help="Convert a molmo2 training manifest into the gaze-serve viewer layout.",
        description=(
            "Each manifest row (a clip segment) becomes one viewer episode with the segment mp4 "
            "as video + a per-frame gaze table + annotation text, plus a manifest.parquet the "
            "`gaze serve` viewer lists. Run `gaze serve --canonical-root <out>` to browse."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    viewer.add_argument("--manifest-root", metavar="PATH", required=True, help="molmo2 build output root (has manifest.jsonl + videos/).")
    viewer.add_argument("--out-root", metavar="PATH", required=True, help="Viewer-layout output root.")
    viewer.set_defaults(func=cmd_curate_viewer_layout)

    join = curate_sub.add_parser(
        "join-manifests",
        help="Concatenate per-run manifests into one joint manifest.jsonl (drop off-dataset strays).",
        description=(
            "Stream-concatenate multiple per-run manifest.jsonl files into ONE joint manifest. "
            "Each --source is PATH[:dataset,dataset] -- keep only those datasets from that file "
            "(omit the :list to keep all). De-dupes clip ids, skips error rows. Constant memory. "
            "Use to merge the non-nymeria and nymeria runs, taking only nymeria from the nym run "
            "and only the 4 real datasets from the nonnym run (dropping sample-default strays)."
        ),
        epilog=(
            "Example:\n"
            "  gaze curate join-manifests \\\n"
            "    --source /path/nonnym/manifest.jsonl:ego-exo4d,egtea,holoassist,hd-epic \\\n"
            "    --source /path/nym/manifest.jsonl:nymeria \\\n"
            "    --out /path/joint/manifest.jsonl"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    join.add_argument("--source", metavar="PATH[:DS,DS]", action="append", required=True, help="A manifest.jsonl, optionally :dataset,dataset to keep only those. Repeatable.")
    join.add_argument("--out", metavar="PATH", required=True, help="Output joint manifest.jsonl path.")
    join.set_defaults(func=cmd_curate_join_manifests)

    mksplit = curate_sub.add_parser(
        "make-splits",
        help="Produce clip-level, per-dataset-stratified train/val split POINTERS from a manifest (+ upload to S3).",
        description=(
            "Read a joint manifest ONCE (streaming, pointer fields only -- no clip data copied) "
            "and emit per-split POINTER files (one {id,dataset,video} per line) that downstream "
            "training joins back to the manifest by id. Sampling is CLIP-LEVEL and PER-DATASET "
            "STRATIFIED: the ratio is applied independently within each dataset then unioned, so "
            "the val split is GUARANTEED to contain clips from every (incl. minority) dataset. "
            "Deterministic given --seed. Optionally uploads to s3://.../splits/<name>/."
        ),
        epilog=(
            "Examples:\n"
            "  gaze curate make-splits --manifest /path/joint/manifest.jsonl --name v1_80_20 \\\n"
            "    --ratios train=0.8,val=0.2 --out-dir /path/splits --upload\n"
            "  gaze curate make-splits --manifest m.jsonl --name s --ratios train=0.7,val=0.15,test=0.15 --seed 42"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mksplit.add_argument("--manifest", metavar="PATH", required=True, help="Joint manifest.jsonl to split. Required.")
    mksplit.add_argument("--name", metavar="NAME", required=True, help="Split name (becomes the S3/dir subfolder). Required.")
    mksplit.add_argument("--ratios", metavar="NAME=FLOAT[,...]", default="train=0.8,val=0.2", help="Split ratios, normalized to sum to 1; default train=0.8,val=0.2.")
    mksplit.add_argument("--seed", metavar="INT", type=int, default=0, help="Deterministic seed; default 0.")
    mksplit.add_argument("--out-dir", metavar="PATH", default="/tmp/gaze_splits", help="Local dir for split pointer files; default /tmp/gaze_splits.")
    mksplit.add_argument("--upload", action="store_true", help="Upload <out-dir>/<name>/ to S3 after writing.")
    mksplit.add_argument("--s3-uri", metavar="URI", default="s3://far-research-internal/colbhben/gaze/splits", help="S3 splits prefix.")
    mksplit.add_argument("--upload-via-host", metavar="HOST", help="Run the S3 sync via ssh on this host (for envs where local AWS is read-only).")
    mksplit.add_argument("--aws-bin", metavar="PATH", default="aws", help="aws binary; default 'aws' (use /snap/bin/aws on the remote).")
    mksplit.add_argument("--dry-run", action="store_true", help="With --upload, print the plan without uploading.")
    mksplit.set_defaults(func=cmd_curate_make_splits)


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


def _sample_episode_id(dataset: str) -> str | None:
    samples_path = Path(__file__).resolve().parents[2] / "recipes" / "_sample_episodes.json"
    if not samples_path.exists():
        return None
    data = json.loads(samples_path.read_text(encoding="utf-8"))
    return (data.get("samples", {}).get(dataset) or {}).get("episode_id")


def _curate_puller(args: argparse.Namespace):
    from .curate import Puller
    return Puller(ssh_host=getattr(args, "ssh_host", None), local_root=getattr(args, "local_root", None))


def cmd_curate_extract(args: argparse.Namespace) -> int:
    from . import curate_readers
    episode = args.episode or _sample_episode_id(args.dataset)
    if not episode:
        print(f"error: no --episode given and no sample episode for {args.dataset}", file=sys.stderr)
        return 1
    puller = _curate_puller(args)
    bundle = curate_readers.extract_episode(args.dataset, episode, puller)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.dataset}.json").write_text(json.dumps(bundle.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(bundle.to_dict(), indent=2, sort_keys=True))
    return 0 if bundle.emitted else 1


def cmd_curate_overlay(args: argparse.Namespace) -> int:
    from .overlay import render_overlay
    episode = args.episode or _sample_episode_id(args.dataset)
    if not episode:
        print(f"error: no --episode given and no sample episode for {args.dataset}", file=sys.stderr)
        return 1
    out = args.out or f"/tmp/gaze_overlays/{args.dataset}.mp4"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    puller = _curate_puller(args)
    result = render_overlay(args.dataset, episode, puller, out, max_seconds=args.max_seconds)
    print(json.dumps(result if isinstance(result, dict) else getattr(result, "__dict__", {"out": out}), indent=2, sort_keys=True, default=str))
    return 0


def cmd_curate_build_smoke(args: argparse.Namespace) -> int:
    from .smoke import build_smoke_manifest
    report = build_smoke_manifest(
        args.extract_dir, args.overlays_dir, args.out_root,
        datasets=parse_set(args.datasets) and sorted(parse_set(args.datasets)),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def cmd_curate_upload_smoke(args: argparse.Namespace) -> int:
    from .smoke import upload_smoke_manifest
    report = upload_smoke_manifest(args.root, ssh_host=args.ssh_host, s3_uri=args.s3_uri, dry_run=args.dry_run)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if (args.dry_run or report.get("returncode") == 0) else 1


def cmd_curate_smoke(args: argparse.Namespace) -> int:
    from .smoke import build_smoke_manifest, upload_smoke_manifest
    report = build_smoke_manifest(args.extract_dir, args.overlays_dir, args.out_root)
    result = {"build": report}
    if args.upload:
        result["upload"] = upload_smoke_manifest(args.out_root, ssh_host=args.ssh_host, s3_uri=args.s3_uri, dry_run=args.dry_run)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.upload and not args.dry_run and result["upload"].get("returncode") not in (0, None):
        return 1
    return 0


def cmd_curate_export_annotations(args: argparse.Namespace) -> int:
    from .classify import export_annotation_spans
    from .overlay import build_episode_data

    slug = args.dataset
    if args.episodes_file:
        ef = json.loads(Path(args.episodes_file).read_text(encoding="utf-8"))
        ids = (ef.get("datasets") or ef).get(slug, [])
    elif args.episode:
        ids = [args.episode]
    else:
        ids = [_sample_episode_id(slug)]
    ids = [e for e in ids if e]
    if not ids:
        print(f"error: no episodes for {slug}", file=sys.stderr)
        return 1

    from .overlay import reconciled_annotations

    puller = _curate_puller(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as fh:
        for ep in ids:
            try:
                data = build_episode_data(slug, ep, puller, reuse=True)
                # Export RECONCILED (video-zero) times so the interesting map is on the
                # same clock the consume side (build-training-manifest) uses.
                bundle = {"annotations": reconciled_annotations(data)}
                rec = export_annotation_spans(bundle, ep)
                fh.write(json.dumps(rec, sort_keys=True) + "\n")
                n += 1
            except Exception as exc:  # noqa
                print(f"warn: {slug}:{ep} export failed: {exc}", file=sys.stderr)
    print(json.dumps({"dataset": slug, "episodes_exported": n, "out": str(out)}, indent=2))
    return 0


def cmd_curate_viewer_layout(args: argparse.Namespace) -> int:
    from .smoke import molmo2_to_viewer_layout

    report = molmo2_to_viewer_layout(args.manifest_root, args.out_root)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def cmd_curate_join_manifests(args: argparse.Namespace) -> int:
    from .splits_manifest import join_manifests

    sources: list[tuple[str, set[str] | None]] = []
    for spec in args.source:
        path, sep, dslist = spec.partition(":")
        keep = {d.strip() for d in dslist.split(",") if d.strip()} if sep else None
        sources.append((path, keep))
    report = join_manifests(sources, args.out)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def cmd_curate_make_splits(args: argparse.Namespace) -> int:
    from .splits_manifest import build_split, upload_splits

    ratios = parse_ratios(args.ratios)
    index = build_split(args.manifest, args.out_dir, name=args.name, ratios=ratios, seed=args.seed)
    result: dict = {"split": index}
    if args.upload:
        result["upload"] = upload_splits(
            args.out_dir, name=args.name, s3_uri=args.s3_uri, aws_bin=args.aws_bin,
            on_remote_host=args.upload_via_host, dry_run=args.dry_run,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.upload and not args.dry_run and result["upload"].get("returncode") not in (0, None):
        return 1
    return 0


def cmd_curate_build_training(args: argparse.Namespace) -> int:
    from .training import build_training_manifest

    # Self-dump all thread stacks if the build wedges (set GAZE_FAULT_TIMEOUT=seconds).
    _ft = os.environ.get("GAZE_FAULT_TIMEOUT")
    if _ft:
        import faulthandler
        faulthandler.dump_traceback_later(float(_ft), repeat=True)

    episodes = None
    if args.episode:
        sel = parse_set(args.datasets)
        if not sel or len(sel) != 1:
            print("error: --episode requires exactly one --datasets slug", file=sys.stderr)
            return 1
        episodes = {next(iter(sel)): args.episode}

    sample_extra = None
    if args.sample:
        data = json.loads(Path(args.sample).read_text(encoding="utf-8"))
        sample_extra = {
            slug: {k: v for k, v in entry.items() if k not in ("note", "episode_id")}
            for slug, entry in (data.get("samples") or {}).items()
        }
        episodes = episodes or {}
        for slug, entry in (data.get("samples") or {}).items():
            episodes.setdefault(slug, entry.get("episode_id"))

    episode_lists = None
    if getattr(args, "episodes_file", None):
        ef = json.loads(Path(args.episodes_file).read_text(encoding="utf-8"))
        episode_lists = ef.get("datasets") or ef

    interesting_maps = {}
    for spec in getattr(args, "interesting_map", []) or []:
        slug, _, path = spec.partition("=")
        if path:
            interesting_maps[slug] = json.loads(Path(path).read_text(encoding="utf-8"))

    puller = _curate_puller(args)
    report = build_training_manifest(
        args.out_root,
        datasets=parse_set(args.datasets) and sorted(parse_set(args.datasets)),
        episodes=episodes,
        episode_lists=episode_lists,
        sample_extra=sample_extra,
        output_format=args.output_format,
        profile=args.profile,
        fps=args.fps,
        num_frames=args.num_frames,
        stride=args.stride,
        resolution=args.resolution,
        temporality=args.temporality,
        window_s=args.window_seconds,
        max_clip_s=args.max_clip_duration_s,
        merge_gap_s=args.merge_gap_s,
        drop_shorter_than_s=args.drop_shorter_than_s,
        min_duration_s=args.min_duration_s,
        max_frames=args.max_frames,
        prompt=args.prompt,
        interesting_maps=interesting_maps,
        workers=getattr(args, "workers", None),
        puller=puller,
        reuse_bundle=not args.no_reuse_bundle,
        reuse_clips=getattr(args, "reuse_clips", False),
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
