#!/usr/bin/env bash
# Full-chain launcher for the Molmo2 gaze SFT run (specialize-then-rehearse).
#
# Pick a hardware profile with --profile l40|h200 (default: l40). The profile sets the
# memory-bound defaults -- docker image, seq_len, device/global batch, and default GPU count
# -- for that card. Any individual flag you pass overrides the profile. --gpus is independent:
# either profile runs single-card (--gpus 1) or full-node (--gpus 8).
#
# One command does the whole chain:
#   1. validate inputs + REQUIRE the user's wandb credentials (no hardcoded keys),
#   2. STAGE data into LOCAL scratch -- /nfs is a strictly READ-ONLY mount, we never write it,
#   3. select the 92%/8% gaze/rehearse specialize mixture (override via --specialize-ratio),
#   4. compose every training-control flag with Molmo2-recommended defaults (all overridable),
#   5. checkpoint to an S3 save_folder by default (native cloud upload; nothing lands on /nfs),
#   6. launch `torchrun --nproc-per-node=$GPUS launch_scripts/sft.py` inside the prebuilt image.
#
# The training OBJECTIVE is flexible (--gaze-objective):
#   first : given the full video, predict the FIRST gaze point (t0).            [default]
#   all   : given the full video, predict ALL per-frame gaze points.
# (Implemented as dataset-construction-time target slicing; the model always sees the full clip.)
#
# Gaze metrics (L2 distance + accuracy@radius) are computed on the held-out gaze val split and
# logged to wandb alongside train loss.
#
# Usage:
#   training/gaze_sft.sh --name <run> --wandb-project <proj> --wandb-entity <team> \
#       --gaze-data-dir <local-dir> --molmo-data-dir <local-dir> [flags...]
#
# Credentials: pass --wandb-key/-project/-entity OR export WANDB_API_KEY/WANDB_PROJECT/
# WANDB_ENTITY before calling. The script FAILS FAST if any are missing. AWS creds for the
# S3 checkpoint upload are read from the host env and passed through to the container.
#
# Flags (every default is overridable; [req] = required):
#   Run / credentials
#     --name NAME              [req] run name; also fills the default S3 save_folder
#     --wandb-key KEY          wandb API key            (default: $WANDB_API_KEY)   [req]
#     --wandb-project PROJ     wandb project            (default: $WANDB_PROJECT)   [req]
#     --wandb-entity ENT       wandb entity/team        (default: $WANDB_ENTITY)    [req]
#     --wandb-base-url URL     self-hosted W&B server URL; REQUIRED for "local-" keys
#                              (default: $WANDB_BASE_URL, else api.wandb.ai)
#     --hf-token TOK           HuggingFace token        (default: $HF_ACCESS_TOKEN)
#   Data (no path may be on /nfs at run time -- stage to local scratch first)
#     --gaze-data-dir DIR      [req] local dir with joint/manifest.jsonl + splits/<name>/
#     --gaze-split-name NAME   split subdir under splits/                  (default: v1_95_05)
#     --molmo-data-dir DIR     [req] local Molmo2-Data rehearsal root
#     --stage-from SRC         copy data out of an /nfs path or s3:// URL into local scratch
#     --local-scratch DIR      where --stage-from lands         (default: /home/ubuntu/gaze-stage)
#     --hf-cache DIR           WRITABLE HuggingFace cache; never /nfs (default: <scratch>/hf-cache)
#   Mixture / objective
#     --mixture NAME           training mixture                       (default: gaze_specialize)
#     --gaze-objective first|all  first=predict t0 point, all=per-frame points   (default: first)
#     --specialize-ratio R     gaze / rehearse ratio                          (default: 0.92)
#   Hardware profile (sets the memory-bound defaults below; individual flags override)
#     --profile l40|h200      l40  = 8xL40 48GB, molmo2:l40s image, seq 8192 / dbatch 1 / gbatch 64
#                             h200 = H200 141GB, stock ghcr image,  seq 16384 / dbatch 2 / gbatch 128
#                                                                                  (default: l40)
#   Model / image
#     --checkpoint PATH        olmo-native starting ckpt   (default: /data/molmo/Molmo2-4B-SFT)
#     --image IMG              docker image                              (default: per --profile)
#   Checkpointing
#     --save-folder URL        output dir, s3:// or local; /nfs rejected
#                              (default: s3://far-research-internal/colbhben/gaze/molmo/runs/<name>)
#     --save-interval N        steps between checkpoint saves                  (default: 2000)
#     --max-duration N         total training steps; e.g. 200 for a smoke   (default: sft.py default)
#   Parallelism / batch (seq-len, device-batch, global-batch default per --profile)
#     --gpus N                 torchrun --nproc-per-node                  (default: per --profile, 8)
#     --cp-degree N            context-parallel degree; >1 auto-sets compile=null (default: 1)
#     --seq-len N              sequence length                              (default: per --profile)
#     --device-batch-size N    per-rank microbatch size                     (default: per --profile)
#     --global-batch-size N    global batch; must divide gpus/cp_degree     (default: per --profile)
#     --num-workers N          dataloader workers                                 (default: 6)
#   Optimizer / schedule
#     --llm-lr LR              LLM learning rate                               (default: 1e-5)
#     --vit-lr LR              ViT learning rate                               (default: 5e-6)
#     --connector-lr LR        connector learning rate                         (default: 5e-6)
#     --warmup N               warmup steps (llm/vit/connector)                 (default: 200)
#     --alpha-f F              scheduler final-LR fraction                      (default: 0.1)
#   Other
#     --dry-run                print the docker run / torchrun command without executing
#     -h, --help               show this help
#     -- <extra...>            everything after -- is passed as raw sft.py dotlist overrides
set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
MOLMO2_DIR="$ROOT/third_party/molmo2"

# ----------------------------------------------------------------------------------------- #
# Defaults (Molmo2 SFT recommended values; every one is overridable via flag).
# ----------------------------------------------------------------------------------------- #
# Hardware profile selects the memory-bound defaults (image, seq_len, device/global batch).
# 'l40'  -> 8xL40 48GB, patched flash-attn image, smaller activations.
# 'h200' -> H200 141GB, stock ghcr image, the Molmo2 reference seq_len/batch.
# Switch with --profile; --gpus is independent (works as 1- or 8-GPU on either). Any knob you
# pass explicitly always wins over the profile. Resolved in apply_profile() after arg parse.
PROFILE="l40"
GPUS=""                                  # empty => profile default (8). --gpus 1 for single-card.
RUN_NAME=""
MIXTURE="gaze_specialize"
GAZE_OBJECTIVE="first"
SPECIALIZE_RATIO="0.92"                  # 92% gaze / 8% rehearse
# Docker image. Empty => profile default. L40 (Ada sm_89) needs the patched flash-attn build:
#   docker build -f Dockerfile.l40s -t molmo2:l40s third_party/molmo2
# H200/B200 (sm_90/100) use the stock ghcr image. MOLMO_IMAGE env, if set, counts as explicit.
IMAGE=${MOLMO_IMAGE:-}

# Released Molmo2 4B VLM checkpoint (olmo-native: config.yaml + model_and_optim/).
CHECKPOINT=${MOLMO_CHECKPOINT:-/data/molmo/Molmo2-4B-SFT}

# Data locations. GAZE_DATA_DIR (LOCAL) must hold joint/manifest.jsonl + splits/<name>/.
# MOLMO_DATA_DIR (LOCAL) holds the rehearsal Molmo2-Data. Neither may be on /nfs at run time.
GAZE_DATA_DIR=${GAZE_DATA_DIR:-}
GAZE_SPLIT_NAME=${GAZE_SPLIT_NAME:-v1_95_05}
MOLMO_DATA_DIR=${MOLMO_DATA_DIR:-}
# Optional: stage GAZE_DATA_DIR from this read-only source (a /nfs path or s3:// URL) into
# a LOCAL scratch dir before training. Leaves /nfs untouched (copies OUT of it).
STAGE_FROM=${STAGE_FROM:-}
LOCAL_SCRATCH=${LOCAL_SCRATCH:-/home/ubuntu/gaze-stage}
# HuggingFace cache. MUST be writable (the tokenizer/model build writes lock files + downloads
# here), so it can NOT live under a read-only MOLMO_DATA_DIR (e.g. /nfs). Empty => defaults to
# <local-scratch>/hf-cache after parse; mounted writable at /hf-cache in the container.
HF_CACHE=${HF_CACHE:-}

# Checkpoints -> S3 by default (native cloud save; never /nfs). {name} filled after parse.
SAVE_FOLDER=""
SAVE_INTERVAL=2000
MAX_DURATION=""                          # steps; short for smoke (e.g. 200). empty=sft.py default

# Training-control knobs. seq_len/device_batch/global_batch are activation-memory bound and
# set by the profile (see apply_profile): L40 (48GB) uses smaller values than H200 (141GB).
# FSDP2 shards params/grads/optim across ranks, but per-rank ACTIVATION memory scales with
# seq_len x device_batch_size and is NOT reduced by adding GPUs. For sequences longer than the
# profile default, prefer --cp-degree (splits the sequence across ranks) over a bigger seq_len.
# Empty => filled by the profile; pass the flag to override.
SEQ_LEN=""
DEVICE_BATCH_SIZE=""
GLOBAL_BATCH_SIZE=""                     # must be divisible by GPUS/CP_DEGREE. empty => profile default
CP_DEGREE=1                              # >1 splits the sequence across ranks; requires compile=null (see launch)
LLM_LR=1e-5
VIT_LR=5e-6
CONNECTOR_LR=5e-6
WARMUP=200
ALPHA_F=0.1
NUM_WORKERS=6

# wandb (REQUIRED). Seed from env; --wandb-* flags override.
WANDB_KEY=${WANDB_API_KEY:-}
WANDB_PROJECT=${WANDB_PROJECT:-}
WANDB_ENTITY=${WANDB_ENTITY:-}
# Base URL of the wandb server. Self-hosted ("local-..." keys) need this pointed at your
# W&B Server; the public cloud (api.wandb.ai) rejects local- keys with HTTP 401. Empty =>
# wandb's default (api.wandb.ai). Set via --wandb-base-url or WANDB_BASE_URL.
WANDB_BASE_URL_=${WANDB_BASE_URL:-}

HF_TOKEN=${HF_ACCESS_TOKEN:-}
DRY_RUN=0
EXTRA=()

# ----------------------------------------------------------------------------------------- #
# Parse args.
# ----------------------------------------------------------------------------------------- #
while [ "$#" -gt 0 ]; do
  case "$1" in
    --profile) PROFILE=$2; shift 2 ;;
    --gpus) GPUS=$2; shift 2 ;;
    --name) RUN_NAME=$2; shift 2 ;;
    --mixture) MIXTURE=$2; shift 2 ;;
    --gaze-objective) GAZE_OBJECTIVE=$2; shift 2 ;;
    --specialize-ratio) SPECIALIZE_RATIO=$2; shift 2 ;;
    --checkpoint) CHECKPOINT=$2; shift 2 ;;
    --image) IMAGE=$2; shift 2 ;;
    --gaze-data-dir) GAZE_DATA_DIR=$2; shift 2 ;;
    --gaze-split-name) GAZE_SPLIT_NAME=$2; shift 2 ;;
    --molmo-data-dir) MOLMO_DATA_DIR=$2; shift 2 ;;
    --stage-from) STAGE_FROM=$2; shift 2 ;;
    --local-scratch) LOCAL_SCRATCH=$2; shift 2 ;;
    --hf-cache) HF_CACHE=$2; shift 2 ;;
    --save-folder) SAVE_FOLDER=$2; shift 2 ;;
    --save-interval) SAVE_INTERVAL=$2; shift 2 ;;
    --max-duration) MAX_DURATION=$2; shift 2 ;;
    --seq-len) SEQ_LEN=$2; shift 2 ;;
    --device-batch-size) DEVICE_BATCH_SIZE=$2; shift 2 ;;
    --global-batch-size) GLOBAL_BATCH_SIZE=$2; shift 2 ;;
    --cp-degree) CP_DEGREE=$2; shift 2 ;;
    --llm-lr) LLM_LR=$2; shift 2 ;;
    --vit-lr) VIT_LR=$2; shift 2 ;;
    --connector-lr) CONNECTOR_LR=$2; shift 2 ;;
    --warmup) WARMUP=$2; shift 2 ;;
    --alpha-f) ALPHA_F=$2; shift 2 ;;
    --num-workers) NUM_WORKERS=$2; shift 2 ;;
    --wandb-key) WANDB_KEY=$2; shift 2 ;;
    --wandb-project) WANDB_PROJECT=$2; shift 2 ;;
    --wandb-entity) WANDB_ENTITY=$2; shift 2 ;;
    --wandb-base-url) WANDB_BASE_URL_=$2; shift 2 ;;
    --hf-token) HF_TOKEN=$2; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --) shift; EXTRA=("$@"); break ;;
    -h|--help) sed -n '2,81p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; exit 2 ;;
  esac
done

die() { echo "ERROR: $*" >&2; exit 1; }

# ----------------------------------------------------------------------------------------- #
# 0. Resolve the hardware profile. Fills ONLY knobs the user left empty, so any explicit
#    flag (or MOLMO_IMAGE env) always wins. --gpus is intentionally profile-independent:
#    either profile runs on 1 or 8 GPUs; the profile only sets its default GPU count and the
#    activation-memory-bound seq_len/device_batch/global_batch.
# ----------------------------------------------------------------------------------------- #
apply_profile() {
  local p_gpus p_image p_seq p_dbs p_gbs
  case "$PROFILE" in
    l40|L40|l40s|L40S)
      # 8xL40 48GB: patched (sm_89) flash-attn image, dialed-down activations.
      p_gpus=8;  p_image="molmo2:l40s"; p_seq=8192;  p_dbs=1; p_gbs=64 ;;
    h200|H200)
      # H200 141GB: stock ghcr image, Molmo2 reference seq_len/batch.
      p_gpus=8;  p_image="ghcr.io/allenai/molmo2:latest"; p_seq=16384; p_dbs=2; p_gbs=128 ;;
    *)
      die "--profile must be 'l40' or 'h200' (got '$PROFILE')." ;;
  esac
  # ${VAR:=default} assigns only when VAR is unset/empty -> explicit flags survive.
  : "${GPUS:=$p_gpus}"
  : "${IMAGE:=$p_image}"
  : "${SEQ_LEN:=$p_seq}"
  : "${DEVICE_BATCH_SIZE:=$p_dbs}"
  : "${GLOBAL_BATCH_SIZE:=$p_gbs}"
}
apply_profile

# ----------------------------------------------------------------------------------------- #
# 1. Validate inputs + REQUIRE wandb creds (fail fast, no hardcoded secrets).
# ----------------------------------------------------------------------------------------- #
[ -n "$RUN_NAME" ] || die "--name is required (used for run name + S3 save_folder)."
[ -f "$MOLMO2_DIR/launch_scripts/sft.py" ] || \
  die "$MOLMO2_DIR not initialized. Run: git submodule update --init --recursive"

[ -n "$WANDB_KEY" ] || die "wandb API key required: pass --wandb-key or export WANDB_API_KEY."
[ -n "$WANDB_PROJECT" ] || die "wandb project required: pass --wandb-project or export WANDB_PROJECT."
[ -n "$WANDB_ENTITY" ] || die "wandb entity required: pass --wandb-entity or export WANDB_ENTITY."
# "local-" keys come from a self-hosted W&B Server and 401 against api.wandb.ai. Require the
# server URL so we don't fail 3 minutes into setup.
case "$WANDB_KEY" in
  local-*) [ -n "$WANDB_BASE_URL_" ] || die "wandb key looks self-hosted ('local-...') but no server URL given; pass --wandb-base-url https://<your-wandb-server> (or export WANDB_BASE_URL)." ;;
esac

case "$GAZE_OBJECTIVE" in first|all) ;; *) die "--gaze-objective must be 'first' or 'all'." ;; esac

# Parallelism sanity: cp_degree must divide the GPU count, and the global batch must divide
# evenly across the data-parallel replicas (world_size / cp_degree).
[ $((GPUS % CP_DEGREE)) -eq 0 ] || die "--gpus ($GPUS) must be divisible by --cp-degree ($CP_DEGREE)."
DP_DEGREE=$((GPUS / CP_DEGREE))
if [ -n "$GLOBAL_BATCH_SIZE" ] && [ $((GLOBAL_BATCH_SIZE % DP_DEGREE)) -ne 0 ]; then
  die "--global-batch-size ($GLOBAL_BATCH_SIZE) must be divisible by data-parallel degree ($DP_DEGREE = gpus/cp_degree)."
fi

# Default S3 save_folder (native cloud save; never writes /nfs).
if [ -z "$SAVE_FOLDER" ]; then
  SAVE_FOLDER="s3://far-research-internal/colbhben/gaze/molmo/runs/$RUN_NAME"
fi
case "$SAVE_FOLDER" in
  /nfs/*) die "refusing to checkpoint to /nfs (read-only mount). Use an s3:// or local scratch path." ;;
esac

# ----------------------------------------------------------------------------------------- #
# 2. Stage data into LOCAL scratch (never write /nfs). If --stage-from is given, copy the
#    joint manifest + splits OUT of the read-only source into LOCAL_SCRATCH and point
#    GAZE_DATA_DIR there. Otherwise GAZE_DATA_DIR must already be a local, populated dir.
# ----------------------------------------------------------------------------------------- #
if [ -n "$STAGE_FROM" ]; then
  dest="$LOCAL_SCRATCH/gaze-data"
  echo ">> staging gaze data from $STAGE_FROM -> $dest (local scratch; /nfs stays read-only)"
  mkdir -p "$dest"
  case "$STAGE_FROM" in
    s3://*)
      command -v aws >/dev/null 2>&1 || command -v /snap/bin/aws >/dev/null 2>&1 || \
        die "aws CLI not found for s3 staging."
      AWS=$(command -v aws || command -v /snap/bin/aws)
      "$AWS" s3 cp "$STAGE_FROM/joint/manifest.jsonl" "$dest/joint/manifest.jsonl"
      "$AWS" s3 cp --recursive "$STAGE_FROM/splits/$GAZE_SPLIT_NAME" "$dest/splits/$GAZE_SPLIT_NAME"
      ;;
    *)
      [ -d "$STAGE_FROM" ] || die "--stage-from '$STAGE_FROM' is not a directory or s3:// URL."
      mkdir -p "$dest/joint" "$dest/splits"
      cp -f "$STAGE_FROM/joint/manifest.jsonl" "$dest/joint/manifest.jsonl"
      cp -rf "$STAGE_FROM/splits/$GAZE_SPLIT_NAME" "$dest/splits/$GAZE_SPLIT_NAME"
      ;;
  esac
  GAZE_DATA_DIR="$dest"
fi

[ -n "$GAZE_DATA_DIR" ] || die "--gaze-data-dir required (local dir with joint/manifest.jsonl + splits/)."
# Reading inputs from /nfs is fine -- the dir is bind-mounted :ro, so the container cannot
# write it. (Only checkpoints must never land on /nfs; see the SAVE_FOLDER guard above.)
# Staging to local scratch is a throughput optimization, not a safety requirement -- opt in
# with --stage-from if NFS read bandwidth bottlenecks the dataloader.
case "$GAZE_DATA_DIR" in
  /nfs/*) echo ">> note: reading gaze data directly from /nfs (mounted :ro). For faster" >&2
          echo "         data loading, stage to local scratch with --stage-from." >&2 ;;
esac
[ -f "$GAZE_DATA_DIR/joint/manifest.jsonl" ] || \
  die "missing $GAZE_DATA_DIR/joint/manifest.jsonl (run: gaze curate join-manifests)."
[ -f "$GAZE_DATA_DIR/splits/$GAZE_SPLIT_NAME/train.jsonl" ] || \
  die "missing $GAZE_DATA_DIR/splits/$GAZE_SPLIT_NAME/train.jsonl (run: gaze curate make-splits)."
[ -f "$GAZE_DATA_DIR/splits/$GAZE_SPLIT_NAME/val.jsonl" ] || \
  die "missing $GAZE_DATA_DIR/splits/$GAZE_SPLIT_NAME/val.jsonl (the gaze eval split)."

[ -n "$MOLMO_DATA_DIR" ] || die "--molmo-data-dir required (local Molmo2-Data rehearsal root)."
# Same as gaze data: reading rehearsal data from /nfs is fine (bind-mounted :ro below).
case "$MOLMO_DATA_DIR" in
  /nfs/*) echo ">> note: reading rehearsal data directly from /nfs (mounted :ro)." >&2 ;;
esac
[ -d "$MOLMO_DATA_DIR" ] || die "MOLMO_DATA_DIR '$MOLMO_DATA_DIR' not found."

# Writable HF cache (default under local scratch; never under the read-only data mount).
[ -n "$HF_CACHE" ] || HF_CACHE="$LOCAL_SCRATCH/hf-cache"
case "$HF_CACHE" in
  /nfs/*) die "HF_CACHE is on /nfs (read-only); the tokenizer build must write here. Use local scratch." ;;
esac
mkdir -p "$HF_CACHE/hub" "$HF_CACHE/datasets" || die "could not create HF cache dir: $HF_CACHE"

# Resolve the checkpoint so it is reachable INSIDE the container. The container only mounts
# /molmo2, /gaze-data and /data/molmo -- an arbitrary host path like /nfs/.../Molmo2-4B-SFT is
# NOT visible. Three cases:
#   s3://...            -> pass through unchanged (molmo loads s3 checkpoints natively).
#   existing host path  -> bind-mount it read-only at /checkpoint and point sft.py there.
#   anything else       -> assume it is already an in-container path (e.g. the default
#                          /data/molmo/Molmo2-4B-SFT, which resolves via the /data/molmo mount).
CKPT_MOUNT=()
case "$CHECKPOINT" in
  s3://*)
    CKPT_ARG="$CHECKPOINT" ;;
  *)
    if [ -e "$CHECKPOINT" ]; then
      CKPT_HOST=$(CDPATH= cd -- "$CHECKPOINT" && pwd)   # canonicalize for the bind mount
      CKPT_MOUNT=( -v "$CKPT_HOST:/checkpoint:ro" )
      CKPT_ARG="/checkpoint"
      echo ">> mounting checkpoint $CKPT_HOST -> /checkpoint (read-only)" >&2
    else
      CKPT_ARG="$CHECKPOINT"   # treat as an in-container path
    fi ;;
esac

# ----------------------------------------------------------------------------------------- #
# 3-5. Compose the sft.py command. Mixture, objective + ratio (env), flags, S3 ckpt.
# ----------------------------------------------------------------------------------------- #
SFT_ARGS=(
  "$CKPT_ARG" "$MIXTURE"
  "--seq_len=$SEQ_LEN"
  "--device_batch_size=$DEVICE_BATCH_SIZE"
  "--cp_degree=$CP_DEGREE"
  "--num_workers=$NUM_WORKERS"
  # OmegaConf dotlist overrides merged on top of the TrainConfig:
  "save_folder=$SAVE_FOLDER"
  "save_overwrite=true"
  "save_interval=$SAVE_INTERVAL"
  "run_name=$RUN_NAME"
  "optimizer.llm_learning_rate=$LLM_LR"
  "optimizer.vit_learning_rate=$VIT_LR"
  "optimizer.connector_learning_rate=$CONNECTOR_LR"
  "scheduler.llm_t_warmup=$WARMUP"
  "scheduler.vit_t_warmup=$WARMUP"
  "scheduler.connector_t_warmup=$WARMUP"
  "scheduler.alpha_f=$ALPHA_F"
)
[ -n "$MAX_DURATION" ] && SFT_ARGS+=( "max_duration=$MAX_DURATION" )
[ -n "$GLOBAL_BATCH_SIZE" ] && SFT_ARGS+=( "global_train_batch_size=$GLOBAL_BATCH_SIZE" )
# Context parallelism is incompatible with torch.compile (sft.py enables compile by default
# and run_trainer.py asserts cp_degree==1 when compiling). Disable compile when CP is on.
[ "$CP_DEGREE" -gt 1 ] && SFT_ARGS+=( "compile=null" )
[ "${#EXTRA[@]}" -gt 0 ] && SFT_ARGS+=( "${EXTRA[@]}" )

echo "=================================================================="
echo " Molmo2 gaze SFT run"
echo "   run name        : $RUN_NAME"
echo "   profile         : $PROFILE  (seq_len $SEQ_LEN, device_batch $DEVICE_BATCH_SIZE, global_batch $GLOBAL_BATCH_SIZE)"
echo "   gpus            : $GPUS  (cp_degree $CP_DEGREE)"
echo "   mixture         : $MIXTURE  (gaze ${SPECIALIZE_RATIO} / rehearse)"
echo "   objective       : $GAZE_OBJECTIVE"
echo "   checkpoint      : $CHECKPOINT  (in-container: $CKPT_ARG)"
echo "   gaze data dir   : $GAZE_DATA_DIR  (split: $GAZE_SPLIT_NAME)"
echo "   molmo data dir  : $MOLMO_DATA_DIR"
echo "   hf cache        : $HF_CACHE  (writable; mounted /hf-cache)"
echo "   save_folder     : $SAVE_FOLDER  (S3 native; /nfs untouched)"
echo "   wandb           : $WANDB_ENTITY/$WANDB_PROJECT${WANDB_BASE_URL_:+  (server: $WANDB_BASE_URL_)}"
echo "   image           : $IMAGE"
echo "=================================================================="

# Resolve the image (skip docker entirely in dry-run so the script runs anywhere).
if [ "$DRY_RUN" -eq 0 ]; then
  command -v docker >/dev/null 2>&1 || die "docker not found on PATH."
  if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo ">> image $IMAGE not local; pulling..."
    docker pull "$IMAGE" || { echo ">> pull failed; building from $MOLMO2_DIR/Dockerfile"; \
      IMAGE="molmo2:local"; docker build -t "$IMAGE" "$MOLMO2_DIR"; }
  fi
fi

# AWS creds for the S3 checkpoint upload are passed through from the host env if present.
AWS_ENVS=()
for v in AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_DEFAULT_REGION AWS_REGION; do
  [ -n "${!v:-}" ] && AWS_ENVS+=( -e "$v=${!v}" )
done

# ----------------------------------------------------------------------------------------- #
# 6. Launch (1xH200 by default). Bind-mount the fork (live code), gaze data, rehearsal data.
#    GAZE_OBJECTIVE / GAZE_SPLIT_NAME / GAZE_SPECIALIZE_RATIO are read by our dataset + mixture.
# ----------------------------------------------------------------------------------------- #
RUN=(
  docker run --rm --gpus all --shm-size=32g
  -v "$MOLMO2_DIR:/molmo2"
  -v "$GAZE_DATA_DIR:/gaze-data:ro"
  -v "$MOLMO_DATA_DIR:/data/molmo:ro"
  -v "$HF_CACHE:/hf-cache"
  ${CKPT_MOUNT[@]+"${CKPT_MOUNT[@]}"}
  -w /molmo2
  -e OLMO_SHARED_FS=1
  # Prepared Molmo2-Data tree (read-only). Disk-backed datasets load_from_disk() from here;
  # the canonical scripts/download_datasets.py layout (torch_datasets/, video_datasets/,
  # multi_image_datasets/) must already exist under it.
  -e MOLMO_DATA_DIR=/data/molmo
  # HF cache -> writable scratch. HF_HOME is the root the hub + datasets caches derive from,
  # so HF-only datasets (e.g. tulu4) and the tokenizer download/cache here, never into the
  # read-only data mount. Belt-and-suspenders: pin the sub-caches explicitly too.
  -e HF_HOME=/hf-cache
  -e HF_HUB_CACHE=/hf-cache/hub
  -e HF_DATASETS_CACHE=/hf-cache/datasets
  -e OMP_NUM_THREADS=8
  -e GAZE_DATA_DIR=/gaze-data
  -e GAZE_OBJECTIVE="$GAZE_OBJECTIVE"
  -e GAZE_SPLIT_NAME="$GAZE_SPLIT_NAME"
  -e GAZE_SPECIALIZE_RATIO="$SPECIALIZE_RATIO"
  -e WANDB_API_KEY="$WANDB_KEY"
  -e WANDB_PROJECT="$WANDB_PROJECT"
  -e WANDB_ENTITY="$WANDB_ENTITY"
  ${AWS_ENVS[@]+"${AWS_ENVS[@]}"}
)
# Point wandb at a self-hosted server when a base URL is given (required for "local-" keys).
[ -n "$WANDB_BASE_URL_" ] && RUN+=( -e WANDB_BASE_URL="$WANDB_BASE_URL_" )
# Pass the HF token under BOTH names the code reads (HF_ACCESS_TOKEN in tokenizer.py +
# most loaders; HF_TOKEN in academic_video_datasets.py), so gated repos authenticate either way.
[ -n "$HF_TOKEN" ] && RUN+=( -e HF_ACCESS_TOKEN="$HF_TOKEN" -e HF_TOKEN="$HF_TOKEN" )
RUN+=(
  "$IMAGE"
  bash -lc "set -e; pip install -e . >/dev/null 2>&1 || true; \
    torchrun --nproc-per-node=$GPUS launch_scripts/sft.py ${SFT_ARGS[*]}"
)

if [ "$DRY_RUN" -eq 1 ]; then
  echo ">> DRY RUN -- command that would execute:"
  printf '%q ' "${RUN[@]}"; echo
  exit 0
fi

exec "${RUN[@]}"
