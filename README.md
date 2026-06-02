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

So the default pipeline writes through regular filesystem paths under
`/nfs/colbhben/gaze/...`, while still recording canonical `s3://...` URIs in
manifests. Copy the example config only if a machine needs to override the
bucket path, mount path, or access mode:

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
or mount. Set `"access_mode": "awscli"` only for environments without the NFS
mount; the default `"access_mode": "file_mount"` uses local file copies/syncs.

### A. Serial Download And Raw Backup

This downloads one selected asset at a time, backs it up under
`unprocessed/`, and writes/uploads a download manifest:

```sh
gaze s3 backup-raw \
  --s3-config configs/s3.json \
  --raw-root .gaze-cache/raw \
  --datasets aea \
  --modalities video,gaze,annotation \
  --manifest-name aea-v1
```

Use `--dry-run` first to inspect the planned file-mount copy operations without
downloading or uploading.

### B. Serial Process And Processed Backup

This pulls one raw partition at a time from `unprocessed/`, rectifies it into
canonical form, uploads the processed episode directory under `processed/`, and
uploads `processed/{profile}/manifest.jsonl`:

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

With the default file-mount mode, no AWS CLI is required for moving objects
between the local cache and mounted bucket. The optional AWS CLI settings in
`configs/s3.example.json` are only used when `"access_mode": "awscli"`.
