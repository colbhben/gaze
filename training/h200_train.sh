#!/usr/bin/env bash
# H200 gaze-SFT harness -- meant to be run INSIDE the molmo2 training container (e.g. as a
# SkyPilot `run:` step, one invocation per node). The gaze repo (with the molmo2 submodule) is
# assumed ALREADY CLONED; the harness locates the repo from its OWN path (it lives at
# <repo>/training/h200_train.sh), so it works wherever the repo was cloned (~/gaze,
# ~/sky_workdir/gaze, ...). Override with WORK_DIR=<path>. It invokes training/gaze_sft.sh
# --no-docker with the recommended H200 params, wiring the torchrun multi-node rendezvous from
# SkyPilot's env so an N-node job Just Works.
#
# Why --no-docker: this harness already runs inside the container, so gaze_sft.sh must NOT
# spin another one -- it runs torchrun directly in the current env (see gaze_sft.sh --no-docker).
#
# Multi-node wiring (read from SkyPilot, all overridable via env):
#   SKYPILOT_NUM_NODES          -> --nnodes
#   SKYPILOT_NODE_RANK          -> --node-rank
#   SKYPILOT_NODE_IPS           -> first line is the head node; used for --rdzv-endpoint
#   SKYPILOT_NUM_GPUS_PER_NODE  -> --gpus (per node)
#   SKYPILOT_TASK_ID            -> default --rdzv-id (stable across the job's nodes)
# Single-node falls back to sensible defaults (1 node, 8 GPU) when these are unset.
#
# Everything below is overridable from the environment so the same script serves smoke +
# full runs. Example (SkyPilot run: block):
#   GAZE_RUN_NAME=gaze-specialize-h200-16gpu MAX_DURATION=20000 bash training/h200_train.sh
set -euo pipefail
die() { echo "ERROR: $*" >&2; exit 1; }

# This script lives at <gaze-repo>/training/h200_train.sh, so by default the gaze repo is the
# parent of the script's own directory -- regardless of WHERE the repo was cloned (~/gaze,
# ~/sky_workdir/gaze, etc.). Resolve it from $0 so the harness always finds itself. Override
# with WORK_DIR=<path> if you want to run a different checkout than the one this script is in.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" && pwd)"

# --------------------------------------------------------------------------------------- #
# Config (env-overridable). Defaults target the full Stage-1 specialize run (gaze-67t.3.1).
# --------------------------------------------------------------------------------------- #
WORK_DIR="${WORK_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"   # gaze repo root (parent of training/)
RDZV_PORT="${RDZV_PORT:-29500}"

# Training params (passed through to gaze_sft.sh; override any from the env).
GAZE_RUN_NAME="${GAZE_RUN_NAME:-gaze-specialize-h200}"
PROFILE="${PROFILE:-h200}"
MIXTURE="${MIXTURE:-gaze_specialize}"
SPECIALIZE_RATIO="${SPECIALIZE_RATIO:-0.92}"
GAZE_OBJECTIVE="${GAZE_OBJECTIVE:-all}"
GAZE_DATA_DIR="${GAZE_DATA_DIR:-/nfs/colbhben/gaze/manifests/full_2hz_min25_max16}"
GAZE_SPLIT_NAME="${GAZE_SPLIT_NAME:-95_05}"
MOLMO_DATA_DIR="${MOLMO_DATA_DIR:-/nfs/colbhben/gaze/molmo/Molmo2-Data}"
CHECKPOINT="${CHECKPOINT:-/nfs/colbhben/gaze/molmo/Molmo2-4B-SFT}"
# Read all HuggingFace data (rehearse datasets, baked .filter() caches, the Qwen3 tokenizer +
# siglip2) from the PRE-STAGED cache under MOLMO_DATA_DIR/huggingface instead of downloading at
# train time. Avoids HF 429s AND the s3-fuse cache-write hang (.filter() result-caches go to
# local /tmp). The cache must already be fully staged on /nfs (it is, for this project). Set
# HF_OFFLINE=0 to fall back to online download into local scratch.
HF_OFFLINE="${HF_OFFLINE:-1}"
# Sequence length. MUST be >= ~11357: the video preprocessor's worst-case output for a
# 128-frame clip is 11357 tokens, and get_output_shapes() hard-errors if seq_len is smaller
# (this is a static config check, independent of the actual clip lengths). The H200 profile
# default of 8192 is too small for video and fails at dataloader build, so we pin 12288 here.
SEQ_LEN="${SEQ_LEN:-12288}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-}"        # empty => derived below from world size
MAX_DURATION="${MAX_DURATION:-20000}"             # full specialize; set small (e.g. 2) to smoke
SAVE_INTERVAL="${SAVE_INTERVAL:-2000}"
# Per-component learning rates (Molmo2 SFT recipe). Set explicitly so the run's LRs are pinned
# here rather than relying on gaze_sft.sh's defaults. ViT/connector are gentler than the LLM so
# the already-trained vision stack + connector adapt slowly while the LLM learns the gaze task.
LLM_LR="${LLM_LR:-1e-5}"
VIT_LR="${VIT_LR:-5e-6}"
CONNECTOR_LR="${CONNECTOR_LR:-5e-6}"
# Gaze inference eval (L2 / acc@radius / valid + prediction videos) cadence, in steps. The
# gaze mixture defaults this to 4 (smoke cadence); a full run wants a coarser stride so the
# eval (autoregressive generation) doesn't dominate wall-clock. Passed as a raw sft.py
# dotlist override below.
INF_EVAL_INTERVAL="${INF_EVAL_INTERVAL:-500}"
# How many held-out gaze episodes to score per eval (sft.py --max_inf_eval_examples). Smokes
# used 4; a real run wants a more stable estimate -> 32. Each is an autoregressive generation,
# so this trades eval wall-clock for metric stability.
EVAL_EXAMPLES="${EVAL_EXAMPLES:-32}"
WANDB_PROJECT="${WANDB_PROJECT:-colbhben-gaze}"
WANDB_ENTITY="${WANDB_ENTITY:-far-wandb}"
WANDB_BASE_URL="${WANDB_BASE_URL:-https://far.wandb.io}"
# WANDB_API_KEY / HF_ACCESS_TOKEN are read from the env by gaze_sft.sh; pass them through.

# --------------------------------------------------------------------------------------- #
# Derive node / GPU / rendezvous topology from SkyPilot (with single-node fallbacks).
# --------------------------------------------------------------------------------------- #
NNODES="${SKYPILOT_NUM_NODES:-1}"
NODE_RANK="${SKYPILOT_NODE_RANK:-0}"
GPUS_PER_NODE="${SKYPILOT_NUM_GPUS_PER_NODE:-8}"
RDZV_ID="${RDZV_ID:-${SKYPILOT_TASK_ID:-$GAZE_RUN_NAME}}"
# Head node = first IP in SKYPILOT_NODE_IPS (newline-separated). Fall back to localhost.
HEAD_IP="$(printf '%s\n' "${SKYPILOT_NODE_IPS:-127.0.0.1}" | head -n1)"
RDZV_ENDPOINT="${RDZV_ENDPOINT:-$HEAD_IP:$RDZV_PORT}"

WORLD=$(( GPUS_PER_NODE * NNODES ))
# Default global batch: 1 packed sequence per GPU (cp_degree 1) -> world size. Scales with
# the node count (8 @ 1 node-of-8 is below the profile default; the H200 profile sets 32 for
# a single node, so prefer the profile/explicit value unless the user overrides here).
if [ -z "$GLOBAL_BATCH_SIZE" ]; then
  # 4 packed sequences per GPU matches the H200 profile (dbatch 4, grad-accum 1): 32 @ 8 GPU,
  # 64 @ 16 GPU, etc. -- i.e. world * 4 / dbatch... but dbatch==gbatch/world here, so use world*4.
  GLOBAL_BATCH_SIZE=$(( WORLD * 4 ))
fi

echo "=================================================================="
echo " H200 gaze-SFT harness"
echo "   repo            : $WORK_DIR (resolved from script location)"
echo "   nodes           : $NNODES (this node rank $NODE_RANK), $GPUS_PER_NODE GPU/node => world $WORLD"
echo "   rendezvous      : id=$RDZV_ID endpoint=$RDZV_ENDPOINT"
echo "   run name        : $GAZE_RUN_NAME  (profile $PROFILE, mixture $MIXTURE @ $SPECIALIZE_RATIO)"
echo "   seq_len         : $SEQ_LEN"
echo "   global batch    : $GLOBAL_BATCH_SIZE   max_duration: $MAX_DURATION"
echo "   learning rates  : llm=$LLM_LR vit=$VIT_LR connector=$CONNECTOR_LR"
echo "   eval            : every $INF_EVAL_INTERVAL steps, $EVAL_EXAMPLES episodes"
echo "   gaze data       : $GAZE_DATA_DIR (split $GAZE_SPLIT_NAME)"
echo "   hf data         : $([ "$HF_OFFLINE" -eq 1 ] && echo "OFFLINE from $MOLMO_DATA_DIR/huggingface (pre-staged)" || echo "online download -> local scratch")"
echo "=================================================================="

# --------------------------------------------------------------------------------------- #
# 1. Sanity-check the repo (resolved from this script's location; override via WORK_DIR).
# --------------------------------------------------------------------------------------- #
[ -f "$WORK_DIR/training/gaze_sft.sh" ] || \
  die "gaze repo not found at $WORK_DIR (expected training/gaze_sft.sh). Override with WORK_DIR=<repo>."
[ -f "$WORK_DIR/third_party/molmo2/launch_scripts/sft.py" ] || \
  die "molmo2 submodule not initialized under $WORK_DIR/third_party/molmo2 (run: git submodule update --init --recursive)."

# --------------------------------------------------------------------------------------- #
# 1b. PixMo baked-path shim. PixMo datasets store each image's path as an ABSOLUTE path that
#     was frozen at download time on the box that built Molmo2-Data (MOLMO_DATA_DIR=/data/molmo),
#     e.g. /data/molmo/torch_datasets/pixmo_images/<hash>. olmo's load_image() opens that path
#     verbatim -- it does NOT re-resolve against the current MOLMO_DATA_DIR. On this cluster the
#     same files live under $MOLMO_DATA_DIR (/nfs/...), so we symlink the frozen root
#     (/data/molmo) -> $MOLMO_DATA_DIR so the baked paths resolve. No-op if /data/molmo already
#     exists (e.g. the build box). Override the frozen root via PIXMO_BAKED_ROOT.
PIXMO_BAKED_ROOT="${PIXMO_BAKED_ROOT:-/data/molmo}"
if [ "$MOLMO_DATA_DIR" != "$PIXMO_BAKED_ROOT" ] && [ ! -e "$PIXMO_BAKED_ROOT" ]; then
  echo ">> linking PixMo baked root $PIXMO_BAKED_ROOT -> $MOLMO_DATA_DIR (frozen abs image paths)"
  mkdir -p "$(dirname "$PIXMO_BAKED_ROOT")"
  ln -s "$MOLMO_DATA_DIR" "$PIXMO_BAKED_ROOT" 2>/dev/null \
    || echo ">> WARN: could not create $PIXMO_BAKED_ROOT symlink (may already exist or perms); continuing"
fi

# --------------------------------------------------------------------------------------- #
# 2. Invoke gaze_sft.sh --no-docker (we are already inside the container). Multi-node flags
#    are only added when NNODES>1; single-node reduces to the plain --gpus form.
# --------------------------------------------------------------------------------------- #
ARGS=(
  --profile "$PROFILE" --no-docker
  --gpus "$GPUS_PER_NODE"
  --name "$GAZE_RUN_NAME"
  --mixture "$MIXTURE" --specialize-ratio "$SPECIALIZE_RATIO" --gaze-objective "$GAZE_OBJECTIVE"
  --gaze-data-dir "$GAZE_DATA_DIR" --gaze-split-name "$GAZE_SPLIT_NAME"
  --molmo-data-dir "$MOLMO_DATA_DIR"
  --checkpoint "$CHECKPOINT"
  --seq-len "$SEQ_LEN"
  --global-batch-size "$GLOBAL_BATCH_SIZE"
  --max-duration "$MAX_DURATION" --save-interval "$SAVE_INTERVAL"
  --llm-lr "$LLM_LR" --vit-lr "$VIT_LR" --connector-lr "$CONNECTOR_LR"
  --wandb-project "$WANDB_PROJECT" --wandb-entity "$WANDB_ENTITY" --wandb-base-url "$WANDB_BASE_URL"
)
if [ "$NNODES" -gt 1 ]; then
  ARGS+=( --nnodes "$NNODES" --node-rank "$NODE_RANK" --rdzv-endpoint "$RDZV_ENDPOINT" --rdzv-id "$RDZV_ID" )
fi
# Offline HF (default): read the pre-staged cache under MOLMO_DATA_DIR/huggingface; no download,
# no cache writes (so the s3-fuse .filter()-cache hang can't occur). gaze_sft.sh defaults
# --hf-cache to MOLMO_DATA_DIR/huggingface under --hf-offline.
if [ "$HF_OFFLINE" -eq 1 ]; then
  ARGS+=( --hf-offline )
fi
# Everything after a single `--` is forwarded by gaze_sft.sh to sft.py: argparse flags
# (e.g. --max_inf_eval_examples) AND OmegaConf dotlist overrides (e.g. inf_eval_interval=).
# We set the eval stride + eval episode count here; append any caller-provided overrides
# ($@, given WITHOUT their own leading `--`) after them.
ARGS+=( -- "--max_inf_eval_examples=$EVAL_EXAMPLES" "inf_eval_interval=$INF_EVAL_INTERVAL" "$@" )

cd "$WORK_DIR"
echo ">> exec training/gaze_sft.sh ${ARGS[*]}"
exec bash training/gaze_sft.sh "${ARGS[@]}"
