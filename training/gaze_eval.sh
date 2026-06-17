#!/usr/bin/env bash
# Standalone gaze EVAL launcher: trained checkpoint -> ALL val episodes -> cached S3 results.
#
# Mirrors training/gaze_sft.sh's launch model (hardware profile, docker/no-docker, AWS-env
# passthrough) but runs launch_scripts/eval_gaze.py on a SINGLE GPU instead of training. It runs
# the native molmo2 gaze eval (GazePointEval: L2 0-100 + acc@5/10/15 + valid) over the held-out
# gaze val split and uploads a self-describing cache the gaze viewer ingests:
#
#   s3://far-research-internal/colbhben/gaze/evals/<run_name>/step<N>/
#       summary.json     aggregate metrics + run/checkpoint slugs + episode count
#       results.jsonl    one record per episode (GT + predicted gaze + per-episode metrics)
#
# The val split comes from GAZE_DATA_DIR (joint/manifest.jsonl + splits/<split>/val.jsonl),
# exactly as gaze_sft.sh sets it up. Stage that data to LOCAL scratch first (never /nfs writes).
#
# Usage:
#   training/gaze_eval.sh --checkpoint s3://.../runs/<run>/step<N>/ \
#       --gaze-data-dir <local-dir> --gaze-split-name <name> --gaze-objective first|all \
#       --bundle-s3-uri s3://.../manifests/<bundle>/ [--profile l40|h200] [flags...]
#
# Flags (every default is overridable; [req] = required):
#   --checkpoint PATH        [req] olmo-native ckpt dir (config.yaml + model_and_optim/); s3:// ok
#   --gaze-data-dir DIR      [req] LOCAL dir with joint/manifest.jsonl + splits/<name>/val.jsonl
#   --gaze-split-name NAME   split subdir under splits/                  (default: v1_95_05)
#   --gaze-objective first|all  match the training objective for this run  (default: first)
#   --bundle-s3-uri URI      [req] S3 bundle root; videos/<dataset>/*.mp4 live under it (the
#                            viewer fetches video from here). e.g.
#                            s3://far-research-internal/colbhben/gaze/manifests/full_2hz_min25_max16/
#   --out-uri URI            output prefix (default: s3://.../evals/<run_name>/step<N>/)
#   --max-examples N         cap episodes for a smoke; -1 = ALL val            (default: -1)
#   --device-batch-size N    per-step batch size                               (default: 2)
#   --max-new-tokens N       generation cap                              (default: task default 1024)
#   --overwrite              re-run even if summary.json already exists at the output prefix
#   --profile l40|h200       selects the docker image (l40=molmo2:l40s, h200=stock ghcr) (default: l40)
#   --image IMG              docker image                                      (default: per --profile)
#   --gpus DEV               GPU to use (docker --gpus arg or CUDA index)      (default: 1)
#   --hf-token TOK           HuggingFace token              (default: $HF_ACCESS_TOKEN)
#   --no-docker              run torchrun directly in the CURRENT env (paths used as-is)
#   --dry-run                print the command without executing
#   -h, --help               show this help
#   -- <extra...>            everything after -- is passed as raw eval_gaze.py dotlist overrides
set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
MOLMO2_DIR="$ROOT/third_party/molmo2"

PROFILE="l40"
IMAGE=${MOLMO_IMAGE:-}
GPUS="1"
NO_DOCKER=0
DRY_RUN=0

CHECKPOINT=""
GAZE_DATA_DIR=${GAZE_DATA_DIR:-}
GAZE_SPLIT_NAME=${GAZE_SPLIT_NAME:-v1_95_05}
GAZE_OBJECTIVE=${GAZE_OBJECTIVE:-first}
BUNDLE_S3_URI=""
OUT_URI=""
MAX_EXAMPLES="-1"
DEVICE_BATCH_SIZE="2"
MAX_NEW_TOKENS=""
OVERWRITE=0
HF_TOKEN=${HF_ACCESS_TOKEN:-}
EXTRA=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    --checkpoint) CHECKPOINT=$2; shift 2 ;;
    --gaze-data-dir) GAZE_DATA_DIR=$2; shift 2 ;;
    --gaze-split-name) GAZE_SPLIT_NAME=$2; shift 2 ;;
    --gaze-objective) GAZE_OBJECTIVE=$2; shift 2 ;;
    --bundle-s3-uri) BUNDLE_S3_URI=$2; shift 2 ;;
    --out-uri) OUT_URI=$2; shift 2 ;;
    --max-examples) MAX_EXAMPLES=$2; shift 2 ;;
    --device-batch-size) DEVICE_BATCH_SIZE=$2; shift 2 ;;
    --max-new-tokens) MAX_NEW_TOKENS=$2; shift 2 ;;
    --overwrite) OVERWRITE=1; shift ;;
    --profile) PROFILE=$2; shift 2 ;;
    --image) IMAGE=$2; shift 2 ;;
    --gpus) GPUS=$2; shift 2 ;;
    --hf-token) HF_TOKEN=$2; shift 2 ;;
    --no-docker) NO_DOCKER=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --) shift; EXTRA=("$@"); break ;;
    -h|--help) sed -n '2,52p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; exit 2 ;;
  esac
done

die() { echo "ERROR: $*" >&2; exit 1; }

# Resolve the hardware profile -> docker image (same mapping as gaze_sft.sh).
case "$PROFILE" in
  l40|L40|l40s|L40S) : "${IMAGE:=molmo2:l40s}" ;;
  h200|H200)         : "${IMAGE:=ghcr.io/allenai/molmo2:latest}" ;;
  *) die "--profile must be 'l40' or 'h200' (got '$PROFILE')." ;;
esac

# Validate inputs.
[ -n "$CHECKPOINT" ] || die "--checkpoint is required (olmo-native ckpt dir; s3:// ok)."
[ -n "$BUNDLE_S3_URI" ] || die "--bundle-s3-uri is required (videos/<dataset>/*.mp4 live under it)."
[ -f "$MOLMO2_DIR/launch_scripts/eval_gaze.py" ] || \
  die "$MOLMO2_DIR not initialized. Run: git submodule update --init --recursive"
case "$GAZE_OBJECTIVE" in first|all) ;; *) die "--gaze-objective must be 'first' or 'all'." ;; esac

[ -n "$GAZE_DATA_DIR" ] || die "--gaze-data-dir required (local dir with joint/manifest.jsonl + splits/)."
case "$GAZE_DATA_DIR" in
  /nfs/*) echo ">> note: reading gaze data directly from /nfs (mounted :ro)." >&2 ;;
esac
[ -f "$GAZE_DATA_DIR/joint/manifest.jsonl" ] || \
  die "missing $GAZE_DATA_DIR/joint/manifest.jsonl."
[ -f "$GAZE_DATA_DIR/splits/$GAZE_SPLIT_NAME/val.jsonl" ] || \
  die "missing $GAZE_DATA_DIR/splits/$GAZE_SPLIT_NAME/val.jsonl (the gaze eval split)."

# Resolve the checkpoint so it is reachable INSIDE the container (mirror gaze_sft.sh).
CKPT_MOUNT=()
case "$CHECKPOINT" in
  s3://*) CKPT_ARG="$CHECKPOINT" ;;
  *)
    if [ "$NO_DOCKER" -eq 1 ]; then
      CKPT_ARG="$CHECKPOINT"
    elif [ -e "$CHECKPOINT" ]; then
      CKPT_HOST=$(CDPATH= cd -- "$CHECKPOINT" && pwd)
      CKPT_MOUNT=( -v "$CKPT_HOST:/checkpoint:ro" )
      CKPT_ARG="/checkpoint"
      echo ">> mounting checkpoint $CKPT_HOST -> /checkpoint (read-only)" >&2
    else
      CKPT_ARG="$CHECKPOINT"
    fi ;;
esac

# Compose the eval_gaze.py args.
EVAL_ARGS=(
  "$CKPT_ARG"
  "--bundle-s3-uri=$BUNDLE_S3_URI"
  "--max-examples=$MAX_EXAMPLES"
  "--device-batch-size=$DEVICE_BATCH_SIZE"
)
[ -n "$OUT_URI" ] && EVAL_ARGS+=( "--out-uri=$OUT_URI" )
[ -n "$MAX_NEW_TOKENS" ] && EVAL_ARGS+=( "--max-new-tokens=$MAX_NEW_TOKENS" )
[ "$OVERWRITE" -eq 1 ] && EVAL_ARGS+=( "--overwrite" )
[ "${#EXTRA[@]}" -gt 0 ] && EVAL_ARGS+=( "${EXTRA[@]}" )

TORCHRUN="torchrun --nproc-per-node=1"

echo "=================================================================="
echo " Molmo2 gaze EVAL run"
echo "   checkpoint      : $CHECKPOINT  (in-container: $CKPT_ARG)"
echo "   gaze data dir   : $GAZE_DATA_DIR  (split: $GAZE_SPLIT_NAME, objective: $GAZE_OBJECTIVE)"
echo "   bundle s3 uri   : $BUNDLE_S3_URI"
echo "   max examples    : $MAX_EXAMPLES  (device batch $DEVICE_BATCH_SIZE)"
echo "   out uri         : ${OUT_URI:-s3://.../evals/<run>/<step>/ (default)}"
echo "   profile / image : $PROFILE / $IMAGE"
echo "=================================================================="

# AWS creds for the S3 read (checkpoint) + write (eval cache), passed through from the host env.
AWS_ENVS=()
for v in AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_DEFAULT_REGION AWS_REGION; do
  [ -n "${!v:-}" ] && AWS_ENVS+=( -e "$v=${!v}" )
done

# ----------------------------------------------------------------------------------------- #
# --no-docker: run torchrun directly in the CURRENT env (paths used as-is).
# ----------------------------------------------------------------------------------------- #
if [ "$NO_DOCKER" -eq 1 ]; then
  export OLMO_SHARED_FS=1
  export GAZE_DATA_DIR="$GAZE_DATA_DIR"
  export GAZE_OBJECTIVE="$GAZE_OBJECTIVE" GAZE_SPLIT_NAME="$GAZE_SPLIT_NAME"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPUS}"
  [ -n "$HF_TOKEN" ] && export HF_ACCESS_TOKEN="$HF_TOKEN" HF_TOKEN="$HF_TOKEN"
  NODOCKER_RUN=( bash -lc "set -e; cd '$MOLMO2_DIR'; pip install -e . >/dev/null 2>&1 || true; \
    $TORCHRUN launch_scripts/eval_gaze.py ${EVAL_ARGS[*]}" )
  if [ "$DRY_RUN" -eq 1 ]; then
    echo ">> DRY RUN (--no-docker) -- command that would execute in the current env:"
    printf '%q ' "${NODOCKER_RUN[@]}"; echo
    exit 0
  fi
  exec "${NODOCKER_RUN[@]}"
fi

# ----------------------------------------------------------------------------------------- #
# Docker launch (single GPU).
# ----------------------------------------------------------------------------------------- #
if [ "$DRY_RUN" -eq 0 ]; then
  command -v docker >/dev/null 2>&1 || die "docker not found on PATH."
  docker image inspect "$IMAGE" >/dev/null 2>&1 || die "docker image '$IMAGE' not found locally; pull/build it first."
fi

RUN=(
  docker run --rm --gpus "device=$GPUS" --shm-size=32g
  -v "$MOLMO2_DIR:/molmo2"
  -v "$GAZE_DATA_DIR:/gaze-data:ro"
  ${CKPT_MOUNT[@]+"${CKPT_MOUNT[@]}"}
  -w /molmo2
  -e OLMO_SHARED_FS=1
  -e GAZE_DATA_DIR=/gaze-data
  -e GAZE_OBJECTIVE="$GAZE_OBJECTIVE"
  -e GAZE_SPLIT_NAME="$GAZE_SPLIT_NAME"
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  ${AWS_ENVS[@]+"${AWS_ENVS[@]}"}
)
[ -n "$HF_TOKEN" ] && RUN+=( -e HF_ACCESS_TOKEN="$HF_TOKEN" -e HF_TOKEN="$HF_TOKEN" )
RUN+=(
  "$IMAGE"
  bash -lc "set -e; pip install -e . >/dev/null 2>&1 || true; \
    $TORCHRUN launch_scripts/eval_gaze.py ${EVAL_ARGS[*]}"
)

if [ "$DRY_RUN" -eq 1 ]; then
  echo ">> DRY RUN -- command that would execute:"
  printf '%q ' "${RUN[@]}"; echo
  exit 0
fi

exec "${RUN[@]}"
