#!/usr/bin/env bash
# Docker-first launcher for Molmo2 VLM gaze SFT (specialize-then-rehearse).
#
# Runs allenai/molmo2 (our fork at third_party/molmo2) inside AllenAI's prebuilt
# image, with the fork bind-mounted live and NFS mounted for checkpoints + data.
# See docs/molmo2_training.md for the full workflow, env vars, and the
# GPU-arch / checkpoint-format caveats.
#
# Usage:
#   training/molmo_train.sh [--gpus N] [--name RUN] [--mixture MIX] \
#       [--checkpoint PATH] [--image IMG] [--debug] [-- <extra sft.py args>]
#
# Examples:
#   # Trainer debug smoke (needs H200/B200 + the olmo-native Molmo2 VLM checkpoint; see caveats):
#   training/molmo_train.sh --gpus 1 --mixture debug --debug --name dbg
#
#   # Real specialize-then-rehearse once gaze-kp2 registers the mixture:
#   training/molmo_train.sh --gpus 8 --mixture gaze_rehearse --name gaze-smoke-01
set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
MOLMO2_DIR="$ROOT/third_party/molmo2"

# Defaults (override via flags / env).
GPUS=${MOLMO_GPUS:-1}
RUN_NAME=${MOLMO_RUN_NAME:-debug}
MIXTURE=${MOLMO_MIXTURE:-debug}
IMAGE=${MOLMO_IMAGE:-ghcr.io/allenai/molmo2:latest}
# Host NFS root that holds the Molmo2-VLM-olmo/ checkpoint, Molmo2-Data/ (rehearsal), runs/.
DATA_ROOT=${MOLMO_DATA_ROOT:-/nfs/colbhben/gaze/molmo}
# Released Molmo2 VLM checkpoint (olmo-native) to start from (container path under /data/molmo).
CHECKPOINT=${MOLMO_CHECKPOINT:-/data/molmo/Molmo2-VLM-olmo}
DEBUG=""
EXTRA=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    --gpus) GPUS=$2; shift 2 ;;
    --name) RUN_NAME=$2; shift 2 ;;
    --mixture) MIXTURE=$2; shift 2 ;;
    --checkpoint) CHECKPOINT=$2; shift 2 ;;
    --image) IMAGE=$2; shift 2 ;;
    --data-root) DATA_ROOT=$2; shift 2 ;;
    --debug) DEBUG="--debug"; shift ;;
    --) shift; EXTRA=("$@"); break ;;
    -h|--help) sed -n '1,30p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [ ! -f "$MOLMO2_DIR/launch_scripts/sft.py" ]; then
  echo "ERROR: $MOLMO2_DIR not initialized. Run: git submodule update --init --recursive" >&2
  exit 1
fi
if [ ! -d "$DATA_ROOT" ]; then
  echo "ERROR: data root $DATA_ROOT not found (expected NFS mount with Molmo2-VLM-olmo/, Molmo2-Data/)." >&2
  exit 1
fi

# Verify the image is available; fall back to building from the fork's Dockerfile.
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "Image $IMAGE not present locally; attempting pull..."
  if ! docker pull "$IMAGE"; then
    echo "Pull failed. Building from $MOLMO2_DIR/Dockerfile (this is slow)..." >&2
    IMAGE="molmo2:local"
    docker build -t "$IMAGE" "$MOLMO2_DIR"
  fi
fi

SAVE_FOLDER="/data/molmo/runs/$RUN_NAME"

# Env per molmo2 README. WANDB_* are passed through if set in the caller's env.
docker run --rm --gpus all \
  --shm-size=32g \
  -v "$MOLMO2_DIR:/molmo2" \
  -v "$DATA_ROOT:/data/molmo" \
  -w /molmo2 \
  -e HF_DATASETS_OFFLINE=1 \
  -e OLMO_SHARED_FS=1 \
  -e MOLMO_DATA_DIR=/data/molmo \
  -e HF_HOME=/data/molmo/huggingface \
  -e OMP_NUM_THREADS=8 \
  ${WANDB_API_KEY:+-e WANDB_API_KEY="$WANDB_API_KEY"} \
  ${HF_ACCESS_TOKEN:+-e HF_ACCESS_TOKEN="$HF_ACCESS_TOKEN"} \
  "$IMAGE" \
  bash -lc "
    set -e
    # Make our fork's code live in the container (no-op if already installed editable).
    pip install -e . >/dev/null 2>&1 || true
    torchrun --nproc-per-node=$GPUS launch_scripts/sft.py \
      $CHECKPOINT $MIXTURE $DEBUG \
      --save_folder=$SAVE_FOLDER --save_overwrite \
      ${EXTRA[*]:-}
  "
