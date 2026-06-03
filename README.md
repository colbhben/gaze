# Gaze Dataset Pipeline

This repository contains a hardware-agnostic CLI for planning, downloading,
rectifying, validating, splitting, and viewing heterogeneous video/gaze
datasets listed in `DATASETS.md`.

The setup command never downloads full datasets by default. Dataset downloads
are explicit, selectable by dataset, modality, and sequence, and are designed
to be resumable and checksum-verified when provider manifests include hashes.

## Quick Start

```sh
./scripts/setup.sh
. .venv/bin/activate
gaze doctor
gaze datasets plan --modalities video,gaze,annotation
gaze rectify --raw-root /path/to/raw --canonical-root /path/to/canonical
gaze validate alignment --canonical-root /path/to/canonical
gaze split create --canonical-root /path/to/canonical --name default
gaze view --canonical-root /path/to/canonical
```

## CLI Help And Options

Run `gaze -h` to see every available top-level command. Run any command with
`-h` to see the options for that specific workflow:

```sh
gaze datasets fetch -h
gaze rectify -h
gaze s3 backup-raw -h
```

Command words are positional: put them after `gaze` in order, for example
`gaze datasets fetch` or `gaze s3 backup-raw`. Options are invoked with long
flags. Flags that take values use a space before the value, for example
`--canonical-root ./canonical`. Boolean flags are present or absent, for
example `--dry-run` or `--progress`. Comma-separated filters do not need spaces:

```sh
gaze datasets plan --datasets aea,hot3d --modalities video,gaze
gaze rectify --raw-root ./raw --canonical-root ./canonical --episodes ep1,ep2
```

Command reference:

| Command | Meaning |
| --- | --- |
| `gaze doctor` | Checks local runtime dependencies and optional features. |
| `gaze datasets plan` | Estimates selected dataset assets before downloading. |
| `gaze datasets verify-links` | Checks documentation and sampled manifest URLs. |
| `gaze datasets fetch` | Downloads selected raw assets to a local raw root. |
| `gaze rectify` | Converts raw episodes into the canonical layout. |
| `gaze validate alignment` | Validates canonical outputs and optional raw-source alignment. |
| `gaze split create` | Creates deterministic train/holdout manifests. |
| `gaze serve` | Starts the local API and browser viewer. |
| `gaze view` | Starts the viewer and opens it in a browser. |
| `gaze s3 layout` | Shows the configured static S3 layout. |
| `gaze s3 backup-raw` | Downloads selected raw assets and backs them up to S3. |
| `gaze s3 process-serial` | Pulls raw partitions from S3, rectifies them, and uploads processed episodes. |
| `gaze s3 create-pull-manifest` | Creates a static S3 pull manifest from a local split. |
| `gaze s3 pull-processed` | Downloads processed episodes from a static S3 pull manifest. |

Common dataset-selection options:

| Option | How to invoke | Meaning |
| --- | --- | --- |
| `--repo-root` | `--repo-root /path/to/repo` | Repository root containing `DATASETS.md` and `download_links`; defaults to the current directory. |
| `--datasets` | `--datasets aea,hot3d` | Limits work to dataset slugs. Known slugs include `aea`, `hot3d`, `nymeria`, `holoassist`, `egtea`, and `ego-exo4d`. Omit it to include all cataloged datasets. |
| `--modalities` | `--modalities video,gaze,annotation` | Limits work to normalized modalities. Available values are `video`, `gaze`, `annotation`, `depth`, and `other`. |
| `--sequences` | `--sequences loc1_script1_seq1_rec1,ep2` | Limits work to exact provider sequence ids from manifests. |
| `--json` | `--json` | Emits JSON instead of the default human-readable table. |

Dataset download options:

| Option | How to invoke | Meaning |
| --- | --- | --- |
| `--raw-root` | `--raw-root ./raw` | Local destination for downloaded raw assets. Required for `datasets fetch`; defaults to `.gaze-cache/raw` for `s3 backup-raw`. |
| `--dry-run` | `--dry-run` | Shows the planned work without downloading, uploading, deleting, or copying files. |
| `--manifest-out` | `--manifest-out selected-assets.json` | Writes the selected asset manifest before fetching. |
| `--workers` | `--workers 8` | Per-asset ranged-download worker count when a source supports byte ranges. |
| `--download-timeout` | `--download-timeout 300` | HTTP timeout in seconds for download requests. |
| `--progress` | `--progress` | Prints periodic download progress to stderr. |
| `--sample-per-dataset` | `--sample-per-dataset 10` | Number of manifest links to sample per selected dataset for `verify-links`. |
| `--timeout` | `--timeout 30` | HTTP timeout in seconds for `verify-links`. |

Rectification and validation options:

| Option | How to invoke | Meaning |
| --- | --- | --- |
| `--raw-root` | `--raw-root ./raw` | Raw input root. Required for `rectify`; optional for `validate alignment` when comparing canonical data back to raw inputs. |
| `--canonical-root` | `--canonical-root ./canonical` | Canonical output or input root. Required for `rectify`, `validate alignment`, `split create`, `serve`, and `view`. |
| `--dataset` | `--dataset aea` | Rectifies only one dataset slug under the raw root. |
| `--episodes` | `--episodes ep1,ep2` | Rectifies only the listed episode ids. |
| `--config` | `--config configs/rectify.local.json` | Loads a JSON rectification config instead of the built-in defaults. |
| `--set` | `--set target_hz=5 --set video.width=392` | Overrides one config value. Repeat the flag for multiple overrides. Dotted keys address nested config groups. |

Common `--set` keys include `profile_name`, `target_hz`, `video.fps`,
`video.width`, `video.height`, `video.format`, `video.codec`,
`video.resize_mode`, `gaze.frequency_hz`, `gaze.coordinates`,
`annotation.frequency_hz`, `depth.enabled`, `depth.frequency_hz`, and
`validation.time_tolerance_s`.

Split options:

| Option | How to invoke | Meaning |
| --- | --- | --- |
| `--name` | `--name demo` | Split name and output filename stem under `<canonical-root>/splits`; defaults to `default`. |
| `--ratios` | `--ratios train=0.8,holdout=0.2` | Comma-separated split names and fractions. |
| `--seed` | `--seed 42` | Random seed for deterministic split assignment. |
| `--mode` | `--mode heterogeneous` | Split strategy. Available values are `heterogeneous` and `homogeneous`. |
| `--include-datasets` | `--include-datasets aea,hot3d` | Includes only the listed dataset slugs. |
| `--include-modalities` | `--include-modalities video,gaze` | Includes only episodes with the listed modalities. |
| `--group-by` | `--group-by dataset` | Metadata field used for homogeneous grouping. |
| `--stratify-by` | `--stratify-by participant` | Optional metadata field used to balance split assignment. |

Viewer options:

| Option | How to invoke | Meaning |
| --- | --- | --- |
| `--host` | `--host 127.0.0.1` | Interface for the local HTTP server; defaults to `127.0.0.1`. |
| `--port` | `--port 8765` | TCP port for the local HTTP server; defaults to `8765`. |
| `--open` | `--open` | Opens the viewer in the default browser. `gaze view` enables this automatically. |

S3 workflow options:

| Option | How to invoke | Meaning |
| --- | --- | --- |
| `--s3-config` | `--s3-config configs/s3.json` | User S3 config JSON. Defaults to `configs/s3.json`. |
| `--manifest-name` | `--manifest-name aea-v1` | Name for the raw download manifest under `manifests/downloads`. |
| `--max-download-bytes` | `--max-download-bytes 50000000000` | Maximum bytes to download in one local batch. |
| `--reserve-bytes` | `--reserve-bytes 10000000000` | Minimum local free-space reserve to keep unused. |
| `--storage-fraction` | `--storage-fraction 0.75` | Maximum fraction of currently free local space to use for one batch. |
| `--include-unknown-size` | `--include-unknown-size` | Allows assets without known byte sizes into a download batch. |
| `--keep-cache` | `--keep-cache` | Retains local cache files after successful upload or processing. |
| `--dataset-workers` | `--dataset-workers aea=48,hot3d=36` | Overrides ranged-download worker counts for specific datasets. |
| `--stream-uploads` | `--stream-uploads` | Lets `aws s3 cp` stream upload progress directly to the terminal. |
| `--partitions` | `--partitions toy:ep1,toy:ep2` | Required for `s3 process-serial`; each entry is `dataset:partition_id`. |
| `--local-cache-root` | `--local-cache-root .gaze-cache` | Overrides the local cache root from the S3 config for `process-serial`. |
| `--split-path` | `--split-path ./canonical/splits/demo.json` | Local split manifest produced by `gaze split create`. |
| `--output` | `--output .gaze-cache/splits/demo.s3.json` | Local output path for a generated S3 pull manifest. |
| `--profile` | `--profile default-10hz` | Processed profile name for pull-manifest creation. |
| `--upload` | `--upload` | Uploads the generated pull manifest to the configured S3 splits prefix. |
| `--pull-manifest` | `--pull-manifest .gaze-cache/splits/demo.s3.json` | Static S3 pull manifest used by `s3 pull-processed`. |
| `--dest-root` | `--dest-root ./canonical_train` | Local destination root for pulled processed episodes. |
| `--split` | `--split train` | Pulls only one split bucket from a pull manifest. |

Canonical tabular outputs are written as Parquet when a Parquet engine
(`pyarrow`) is installed. On bare machines, the same schemas are written as
explicit JSONL fallback files and referenced from `episode.json`, so alignment
validation and the viewer still work.

## Local Environment

The intended local setup path is:

```sh
./scripts/setup.sh --extras parquet
. .venv/bin/activate
```

That creates `.venv`, installs the editable `gaze` CLI, and installs the
runtime Parquet stack (`pandas` + `pyarrow`) so canonical table files can be
written as real Parquet. The script then runs `gaze doctor` to show whether
`ffmpeg`, `ffprobe`, and Parquet support are available.

For development and tests, use:

```sh
./scripts/setup.sh --extras dev
. .venv/bin/activate
python -m unittest discover -s tests -v
```

For constrained machines where Python packages cannot be downloaded yet:

```sh
./scripts/setup.sh --extras none
```

The core CLI will still work, but table outputs use `.parquet.jsonl` fallback
files until `pyarrow` is installed.

`ffmpeg` is a system dependency rather than a Python package. Install it with
the platform package manager, for example `brew install ffmpeg`,
`apt-get install ffmpeg`, or `conda install -c conda-forge ffmpeg`.

## S3-Backed Workflow

The default S3-backed path is:

```text
s3://far-research-internal/colbhben/gaze
```

The environment is expected to mount `s3://far-research-internal` at `/nfs`:

```yaml
file_mounts:
  /nfs:
    source: s3://far-research-internal
```

The pipeline uses that mount for access/reads, but uploads are intentionally
performed with `aws s3` commands to the canonical S3 path. Copy the example
config only if a machine needs to override the bucket path, mount path, or AWS
settings:

```sh
cp configs/s3.example.json configs/s3.json
```

The configured `bucket_uri` is the root of a static layout:

```text
s3://far-research-internal/colbhben/gaze/
  unprocessed/{dataset}/{partition}/{asset_key}/{filename}
  processed/{profile}/{dataset}/{episode_id}/...
  processed/{profile}/manifest.jsonl
  manifests/downloads/{name}.json
  splits/{name}.s3.json
```

`configs/s3.json` is ignored by git so each user can point at their own bucket
or mount. The default `"upload_mode": "awscli"` uses `aws s3 cp/sync` for
uploads, while `"access_mode": "file_mount"` maps `s3://far-research-internal`
paths to `/nfs`.

### A. Serial Download And Raw Backup

This detects free space on `--raw-root`, rejects roots under `/nfs`, selects
the largest set of known-size assets that fit the local storage budget, then
downloads to the local root and uploads each asset under `unprocessed/` with
`aws s3 cp`:

```sh
gaze s3 backup-raw \
  --s3-config configs/s3.json \
  --raw-root .gaze-cache/raw \
  --datasets aea \
  --modalities video,gaze,annotation \
  --manifest-name aea-v1
```

Use `--dry-run` first to inspect the planned local downloads and AWS uploads without
downloading or uploading. Use `--max-download-bytes`, `--reserve-bytes`, and
`--storage-fraction` to tune the local fitting logic.

### B. Serial Process And Processed Backup

This accesses one raw partition at a time through `/nfs`, copies it into the
local cache, rectifies it into canonical form, uploads the processed episode
directory under `processed/` with `aws s3 sync`, and uploads
`processed/{profile}/manifest.jsonl`:

```sh
gaze s3 process-serial \
  --s3-config configs/s3.json \
  --partitions toy:ep1,toy:ep2 \
  --config configs/rectify.json
```

Partitions are written as `dataset:partition_id`. The default profile is
`default-10hz` unless changed in `configs/s3.json` or the rectification config.
By default, local per-partition cache directories are removed after upload to
keep disk pressure low; pass `--keep-cache` to retain them for debugging.

### C. Split Manifests And Pulling Processed Data

Create a split from a local processed manifest, then convert it into a static
S3 pull manifest:

```sh
gaze split create \
  --canonical-root .gaze-cache/processed/default-10hz \
  --name demo \
  --ratios train=0.8,holdout=0.2

gaze s3 create-pull-manifest \
  --s3-config configs/s3.json \
  --split-path .gaze-cache/processed/default-10hz/splits/demo.json \
  --output .gaze-cache/splits/demo.s3.json \
  --upload
```

Then another machine can pull only the split it needs:

```sh
gaze s3 pull-processed \
  --s3-config configs/s3.json \
  --pull-manifest .gaze-cache/splits/demo.s3.json \
  --split train \
  --dest-root ./canonical_train
```

Processed data access uses `/nfs`, so the viewer can point directly at the
mounted processed prefix:

```sh
gaze serve --canonical-root /nfs/colbhben/gaze/processed/default-10hz
```

Uploads require the AWS CLI because the canonical upload path is always the
`s3://far-research-internal/colbhben/gaze/...` URI.
