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
