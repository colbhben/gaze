#!/usr/bin/env bash
set -euo pipefail

RAW_ROOT="${RAW_ROOT:-.gaze-cache/raw}"
MANIFEST_OUT="${MANIFEST_OUT:-.gaze-cache/sumedhso-L40S/raw_dataset_manifest.json}"
LOCAL_MANIFEST_COPY="${LOCAL_MANIFEST_COPY:-.gaze-cache/raw_dataset_manifest.json}"
POLL_SECONDS="${POLL_SECONDS:-60}"
STABLE_CHECKS="${STABLE_CHECKS:-2}"
DATASETS="${DATASETS:-}"
MODALITIES="${MODALITIES:-}"
SEQUENCES="${SEQUENCES:-}"

cmd=(
  gaze datasets watch-manifest
  --raw-root "$RAW_ROOT"
  --manifest-out "$MANIFEST_OUT"
  --poll-seconds "$POLL_SECONDS"
  --stable-checks "$STABLE_CHECKS"
)

if [[ -n "$DATASETS" ]]; then
  cmd+=(--datasets "$DATASETS")
fi
if [[ -n "$MODALITIES" ]]; then
  cmd+=(--modalities "$MODALITIES")
fi
if [[ -n "$SEQUENCES" ]]; then
  cmd+=(--sequences "$SEQUENCES")
fi

"${cmd[@]}"

if [[ -n "$LOCAL_MANIFEST_COPY" ]]; then
  if [[ "$LOCAL_MANIFEST_COPY" == *:* ]]; then
    scp "$MANIFEST_OUT" "$LOCAL_MANIFEST_COPY"
  else
    mkdir -p "$(dirname "$LOCAL_MANIFEST_COPY")"
    cp "$MANIFEST_OUT" "$LOCAL_MANIFEST_COPY"
  fi
fi
