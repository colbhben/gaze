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
#     --wandb-project PROJ     wandb project                       (default: colbhben-gaze)
#     --wandb-entity ENT       wandb entity/team                   (default: far-wandb)
#     --wandb-base-url URL     self-hosted W&B server URL (required for "local-" keys)
#                              (default: https://far.wandb.io)
#     --hf-token TOK           HuggingFace token        (default: $HF_ACCESS_TOKEN)
#   Data (no path may be on /nfs at run time -- stage to local scratch first)
#     --gaze-data-dir DIR      [req] local dir with joint/manifest.jsonl + splits/<name>/
#     --gaze-split-name NAME   split subdir under splits/                  (default: v1_95_05)
#     --molmo-data-dir DIR     [req] local Molmo2-Data rehearsal root
#     --stage-from SRC         copy data out of an /nfs path or s3:// URL into local scratch
#     --local-scratch DIR      where --stage-from lands         (default: /home/ubuntu/gaze-stage)
#     --hf-cache DIR           WRITABLE HuggingFace cache; never /nfs (default: <scratch>/hf-cache)
#     --hf-offline             stage a pre-built HF cache (only the dataset repos the mix uses
#                              + Qwen3 tokenizer + siglip2, ~13 GB) into LOCAL HF_CACHE, then
#                              run with HF_HUB_OFFLINE=1 (no download at train time, no 429).
#                              Requires --hf-stage-from <s3:// or /nfs path>.
#     --hf-stage-from SRC      source for --hf-offline staging; an s3:// URL or local path that
#                              contains hub/ + datasets/ subdirs (e.g.
#                              s3://far-research-internal/.../Molmo2-Data/huggingface).
#   Mixture / objective
#     --mixture NAME           training mixture                       (default: gaze_specialize)
#     --gaze-objective first|all  first=predict t0 point, all=per-frame points   (default: first)
#     --specialize-ratio R     gaze / rehearse ratio                          (default: 0.92)
#   Hardware profile (sets the memory-bound defaults below; individual flags override)
#     --profile l40|h200      l40  = 8xL40 48GB, molmo2:l40s image, seq 8192 / dbatch 1 / gbatch 64
#                             h200 = H200 141GB, stock ghcr image,  seq 8192 / dbatch 4 / gbatch 32
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
#     --gpus N                 PER-NODE GPU count (torchrun --nproc-per-node) (default: per --profile, 8)
#     --nnodes N               total nodes; >1 emits a multi-node torchrun rendezvous (default: 1)
#     --node-rank R            this node's rank in [0, nnodes) (default: 0; set per node by your scheduler)
#     --rdzv-endpoint HOST:PORT  head node's rendezvous addr:port (REQUIRED when --nnodes>1)
#     --rdzv-id ID             rendezvous run id (default: the run name)
#     --cp-degree N            context-parallel degree; >1 auto-sets compile=null (default: 1)
#     --seq-len N              sequence length                              (default: per --profile)
#     --device-batch-size N    per-rank microbatch size                     (default: per --profile)
#     --global-batch-size N    global batch; must divide (gpus*nnodes)/cp_degree (default: per --profile)
#     --num-workers N          dataloader workers                                 (default: 6)
#   Optimizer / schedule
#     --llm-lr LR              LLM learning rate                               (default: 1e-5)
#     --vit-lr LR              ViT learning rate                               (default: 5e-6)
#     --connector-lr LR        connector learning rate                         (default: 5e-6)
#     --warmup N               warmup steps (llm/vit/connector)                 (default: 200)
#     --alpha-f F              scheduler final-LR fraction                      (default: 0.1)
#   Other
#     --no-docker              run torchrun directly in the CURRENT env (do NOT auto-spin the
#                              container); use when you started the molmo2 container yourself.
#                              Paths used as-is (no container remap); env exported into the shell.
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
# 'h200' -> H200 141GB, stock ghcr image, seq 8192 / dbatch 4 (gaze-clip-sized, no CP needed).
# Switch with --profile; --gpus is independent (works as 1- or 8-GPU on either). Any knob you
# pass explicitly always wins over the profile. Resolved in apply_profile() after arg parse.
PROFILE="l40"
GPUS=""                                  # PER-NODE GPU count. empty => profile default (8). --gpus 1 for single-card.
# Multi-node (e.g. 16 GPU = 2x 8-GPU H200 nodes). Defaults give single-node behavior, so
# existing single-node usage is unchanged. SkyPilot (or your scheduler) sets these per node;
# molmo2's trainer is already multi-node aware (reads torchrun's RANK/WORLD_SIZE/MASTER_*).
NNODES=1                                 # total nodes; >1 => emit multi-node torchrun rendezvous
NODE_RANK=0                              # this node's rank in [0, NNODES)
RDZV_ENDPOINT=""                         # head node "HOST:PORT" for rendezvous (required when NNODES>1)
RDZV_ID=""                               # rendezvous run id; empty => defaults to the run name
# --no-docker: run torchrun directly in the CURRENT environment instead of auto-spinning the
# container. Use this when you have already entered/started the molmo2 container (or set up a
# matching env) yourself and just want the launch composed + executed. Paths are used as-is
# (no /molmo2//gaze-data//data/molmo//checkpoint remap) and env is exported into this shell.
NO_DOCKER=0
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
# HuggingFace cache. ALWAYS a WRITABLE local dir (default <local-scratch>/hf-cache). The HF
# datasets that .filter()/.map() (mantis, tulu4, ...) write a cache-<hash>.arrow back INTO the
# dataset dir, so HF_CACHE can never be a read-only mount (e.g. /nfs). Two modes differ only in
# how the cache is populated:
#   ONLINE (default):  tokenizer/model/datasets download into HF_CACHE on first use.
#   OFFLINE (--hf-offline): STAGE a pre-built cache (only the repos the mix uses) from
#     --hf-stage-from into HF_CACHE up-front, prune stale filter-caches, then HF_HUB_OFFLINE=1
#     -- no train-time download (no 429), but .filter() can still write locally. See below.
HF_CACHE=${HF_CACHE:-}
# Offline HF: 1 => stage the pre-built HF cache (datasets/ + hub/ for the mix's repos +
# tokenizer + siglip2) from --hf-stage-from into local writable scratch, then export
# HF_HUB_OFFLINE=1 + HF_DATASETS_OFFLINE=1 (no download at train time, no 429 risk).
# We do NOT use HF_DATASETS_DISABLE_CACHING because .filter() still writes a cache file
# into the dataset dir; on /nfs that fails with PermissionError, so we keep the cache local.
HF_OFFLINE=${HF_OFFLINE:-0}
# Source of the pre-built HF cache (s3:// URL or local dir). Required when --hf-offline.
HF_STAGE_FROM=${HF_STAGE_FROM:-}

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

# wandb. Project/entity/base-url default to the FAR self-hosted server; --wandb-* flags or
# env override. The API KEY is still required (no secret hardcoded): pass --wandb-key or
# export WANDB_API_KEY.
WANDB_KEY=${WANDB_API_KEY:-}
WANDB_PROJECT=${WANDB_PROJECT:-colbhben-gaze}
WANDB_ENTITY=${WANDB_ENTITY:-far-wandb}
# Base URL of the wandb server. Self-hosted ("local-..." keys) need this pointed at the
# W&B Server; the public cloud (api.wandb.ai) rejects local- keys with HTTP 401.
WANDB_BASE_URL_=${WANDB_BASE_URL:-https://far.wandb.io}

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
    --nnodes) NNODES=$2; shift 2 ;;
    --node-rank) NODE_RANK=$2; shift 2 ;;
    --rdzv-endpoint) RDZV_ENDPOINT=$2; shift 2 ;;
    --rdzv-id) RDZV_ID=$2; shift 2 ;;
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
    --hf-offline) HF_OFFLINE=1; shift ;;
    --hf-stage-from) HF_STAGE_FROM=$2; shift 2 ;;
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
    --no-docker) NO_DOCKER=1; shift ;;
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
      # H200 141GB: stock ghcr image. seq 8192 x dbatch 4 = 32768 tok-units/GPU (same activation
      # budget as the Molmo2 16384x2 reference) but sized to our 2hz gaze clips: 8192 packs
      # several avg clips and still fits the 16s/32-frame tail (32 frames x ~83 tok ~= 2.7K) in
      # one sequence, so no CP needed. gbatch 32 = 8 GPU x dbatch 4 (grad-accum 1) on ONE node;
      # for multi-node scale gbatch with the WORLD GPU count -- e.g. 64 @ 16 GPU (2 nodes:
      # --gpus 8 --nnodes 2), 128 @ 32 GPU. See bead gaze-67t.2.
      p_gpus=8;  p_image="ghcr.io/allenai/molmo2:latest"; p_seq=8192; p_dbs=4; p_gbs=32 ;;
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

# Multi-node sanity: NNODES>=1, NODE_RANK in range, and a rendezvous endpoint when multi-node.
[ "$NNODES" -ge 1 ] || die "--nnodes ($NNODES) must be >= 1."
[ "$NODE_RANK" -ge 0 ] && [ "$NODE_RANK" -lt "$NNODES" ] || \
  die "--node-rank ($NODE_RANK) must be in [0, --nnodes=$NNODES)."
if [ "$NNODES" -gt 1 ] && [ -z "$RDZV_ENDPOINT" ]; then
  die "--rdzv-endpoint HOST:PORT is required when --nnodes ($NNODES) > 1 (the head node's addr:port)."
fi

# Parallelism sanity: cp_degree must divide the *world* GPU count (gpus-per-node x nnodes),
# and the global batch must divide evenly across the data-parallel replicas (world / cp_degree).
WORLD=$((GPUS * NNODES))
[ $((WORLD % CP_DEGREE)) -eq 0 ] || \
  die "world GPUs ($WORLD = --gpus $GPUS x --nnodes $NNODES) must be divisible by --cp-degree ($CP_DEGREE)."
DP_DEGREE=$((WORLD / CP_DEGREE))
if [ -n "$GLOBAL_BATCH_SIZE" ] && [ $((GLOBAL_BATCH_SIZE % DP_DEGREE)) -ne 0 ]; then
  die "--global-batch-size ($GLOBAL_BATCH_SIZE) must be divisible by data-parallel degree ($DP_DEGREE = world/cp_degree = $WORLD/$CP_DEGREE)."
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

# HF cache. ONLINE: writable, never /nfs (tokenizer/model build writes here).
# OFFLINE (--hf-offline + --hf-stage-from): stage the pre-built HF cache from an S3 URL or
# /nfs path into LOCAL writable scratch, then run with HF_HUB_OFFLINE=1. We MUST own the
# cache locally because HF datasets .filter()/.map() write a cache-<hash>.arrow back into
# the dataset's own cache dir (datasets/<repo>/.../cache-*.arrow) -- on this cluster /nfs
# is read-only, so a /nfs HF_HOME fails with PermissionError. We only stage the dataset
# repos the mix actually uses + the Qwen3 tokenizer + siglip2 (~13 GB total).
HF_CACHE="${HF_CACHE:-$LOCAL_SCRATCH/hf-cache}"
case "$HF_CACHE" in
  /nfs/*) die "HF_CACHE on /nfs is read-only here; .filter()/.map() write back into the dataset dir. Use local scratch (e.g. $LOCAL_SCRATCH/hf-cache)." ;;
esac
mkdir -p "$HF_CACHE/hub" "$HF_CACHE/datasets" || die "could not create HF cache dir: $HF_CACHE"

if [ "$HF_OFFLINE" -eq 1 ]; then
  [ -n "$HF_STAGE_FROM" ] || die "--hf-offline requires --hf-stage-from <s3:// URL or /nfs path with hub/+datasets/>"
  # Repos the gaze_specialize mix actually loads (HF datasets + tokenizer + siglip2). Keep this
  # tight -- the staged Molmo2-Data HF cache holds 78 dataset repos but the mix uses 6.
  HF_REPOS=(
    "datasets/allenai___molmo2-vqa2-2014"
    "datasets/allenai___molmo2-okvqa"
    "datasets/HuggingFaceM4___chart_qa"
    "datasets/allenai___molmo2-a-ok-vqa"
    "datasets/allenai___molmo2-mantis-instruct-nlvr2"
    "datasets/allenai___molmo2-tulu4-classified"
    "hub/datasets--allenai--molmo2-vqa2-2014"
    "hub/datasets--allenai--molmo2-okvqa"
    "hub/datasets--HuggingFaceM4--ChartQA"
    "hub/datasets--allenai--molmo2-a-ok-vqa"
    "hub/datasets--allenai--molmo2-mantis-instruct-nlvr2"
    "hub/datasets--allenai--molmo2-tulu4-classified"
    "hub/models--Qwen--Qwen3-4B-Instruct-2507"
    "hub/models--google--siglip2-so400m-patch14-384"
  )
  echo ">> staging HF cache from $HF_STAGE_FROM -> $HF_CACHE (writable; ${#HF_REPOS[@]} repos)"
  case "$HF_STAGE_FROM" in
    s3://*)
      command -v aws >/dev/null 2>&1 || command -v /snap/bin/aws >/dev/null 2>&1 || die "aws CLI required for s3 staging"
      AWS=$(command -v aws || command -v /snap/bin/aws)
      # Tune for parallelism on the H200 nodes.
      "$AWS" configure set s3.max_concurrent_requests 32 || true
      stage_t0=$(date +%s)
      for r in "${HF_REPOS[@]}"; do
        echo "   -> $r"
        "$AWS" s3 cp --recursive --quiet "$HF_STAGE_FROM/$r" "$HF_CACHE/$r" || die "stage failed: $r"
      done
      echo ">> HF stage complete in $(( $(date +%s) - stage_t0 ))s"
      ;;
    *)
      [ -d "$HF_STAGE_FROM" ] || die "--hf-stage-from '$HF_STAGE_FROM' is not s3:// or a local dir"
      stage_t0=$(date +%s)
      for r in "${HF_REPOS[@]}"; do
        [ -d "$HF_STAGE_FROM/$r" ] || die "missing in stage source: $r"
        echo "   -> $r"
        mkdir -p "$HF_CACHE/$(dirname "$r")"
        cp -r "$HF_STAGE_FROM/$r" "$HF_CACHE/$r" || die "stage failed: $r"
      done
      echo ">> HF stage complete in $(( $(date +%s) - stage_t0 ))s"
      ;;
  esac
  # Prune any pre-existing .filter()/.map() result-caches + temp files from the staged
  # datasets/. These cache-<hash>.arrow files are tied to the EXACT datasets/python/pyarrow
  # version, so a cache baked elsewhere never matches this container -- and some staged copies
  # are 0-byte/partial (from interrupted bakes on the source), which makes .filter() die with
  # "ArrowInvalid: Tried reading schema message, was null or length 0". Removing them forces a
  # clean local recompute (~seconds), which is correct and writable here.
  pruned=$(find "$HF_CACHE/datasets" -type f \( -name "cache-*.arrow" -o -name "tmp*" \) -print -delete 2>/dev/null | wc -l | tr -d ' ')
  echo ">> pruned $pruned stale filter-cache/tmp file(s) from staged datasets (will recompute locally)"
fi

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
    if [ "$NO_DOCKER" -eq 1 ]; then
      # No container: use the checkpoint path as-is (no /checkpoint bind-mount remap).
      CKPT_ARG="$CHECKPOINT"
    elif [ -e "$CHECKPOINT" ]; then
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
if [ "$HF_OFFLINE" -eq 1 ]; then
  echo "   hf cache        : $HF_CACHE  (writable, OFFLINE; staged from $HF_STAGE_FROM)"
else
  echo "   hf cache        : $HF_CACHE  (writable; mounted /hf-cache)"
fi
echo "   save_folder     : $SAVE_FOLDER  (S3 native; /nfs untouched)"
echo "   wandb           : $WANDB_ENTITY/$WANDB_PROJECT${WANDB_BASE_URL_:+  (server: $WANDB_BASE_URL_)}"
echo "   image           : $IMAGE"
echo "=================================================================="

# Resolve the image. Skipped entirely in dry-run (so the script runs anywhere) and in
# --no-docker mode (we run torchrun directly in the current env; there is no container to
# pull/build and docker need not even be installed).
if [ "$DRY_RUN" -eq 0 ] && [ "$NO_DOCKER" -eq 0 ]; then
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

# torchrun launch spec, shared by both the docker and --no-docker paths. Single-node
# (NNODES=1): just --nproc-per-node (unchanged). Multi-node: add c10d rendezvous so all nodes
# join via the head node's endpoint. molmo2's trainer reads RANK/WORLD_SIZE/MASTER_* from
# torchrun -- no molmo2 changes needed.
if [ "$NNODES" -gt 1 ]; then
  TORCHRUN="torchrun --nnodes=$NNODES --node-rank=$NODE_RANK --nproc-per-node=$GPUS \
    --rdzv-backend=c10d --rdzv-endpoint=$RDZV_ENDPOINT --rdzv-id=${RDZV_ID:-$RUN_NAME}"
else
  TORCHRUN="torchrun --nproc-per-node=$GPUS"
fi

# ----------------------------------------------------------------------------------------- #
# 6a. --no-docker launch: run torchrun directly in the CURRENT environment. The user has
#     already started the molmo2 container (or an equivalent env) and just wants the composed
#     launch. Paths are used as-is (no container remap); env is exported into this shell.
#     `cd` into the molmo2 dir so launch_scripts/sft.py + the editable install resolve.
# ----------------------------------------------------------------------------------------- #
if [ "$NO_DOCKER" -eq 1 ]; then
  export OLMO_SHARED_FS=1
  export MOLMO_DATA_DIR="$MOLMO_DATA_DIR"
  export HF_HOME="$HF_CACHE" HF_HUB_CACHE="$HF_CACHE/hub" HF_DATASETS_CACHE="$HF_CACHE/datasets"
  # Offline: cache is fully staged locally (writable); no HF download at train time.
  # We do NOT set HF_DATASETS_DISABLE_CACHING -- .filter() must be allowed to write its
  # cache into the (now writable) dataset dir; disabling caching does NOT prevent the
  # write (it just means the result is not reused on a later run).
  if [ "$HF_OFFLINE" -eq 1 ]; then
    export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
  fi
  export OMP_NUM_THREADS=8
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export NCCL_TIMEOUT_MINUTES="${NCCL_TIMEOUT_MINUTES:-20}"
  export GAZE_DATA_DIR="$GAZE_DATA_DIR"
  export GAZE_OBJECTIVE="$GAZE_OBJECTIVE" GAZE_SPLIT_NAME="$GAZE_SPLIT_NAME" GAZE_SPECIALIZE_RATIO="$SPECIALIZE_RATIO"
  export WANDB_API_KEY="$WANDB_KEY" WANDB_PROJECT="$WANDB_PROJECT" WANDB_ENTITY="$WANDB_ENTITY"
  [ -n "$WANDB_BASE_URL_" ] && export WANDB_BASE_URL="$WANDB_BASE_URL_"
  [ -n "$HF_TOKEN" ] && export HF_ACCESS_TOKEN="$HF_TOKEN" HF_TOKEN="$HF_TOKEN"
  # AWS creds + NIC pins are inherited from the current shell as-is; nothing to remap.
  NODOCKER_RUN=( bash -lc "set -e; cd '$MOLMO2_DIR'; pip install -e . >/dev/null 2>&1 || true; \
    $TORCHRUN launch_scripts/sft.py ${SFT_ARGS[*]}" )
  if [ "$DRY_RUN" -eq 1 ]; then
    echo ">> DRY RUN (--no-docker) -- command that would execute in the current env:"
    printf '%q ' "${NODOCKER_RUN[@]}"; echo
    exit 0
  fi
  exec "${NODOCKER_RUN[@]}"
fi

# ----------------------------------------------------------------------------------------- #
# 6. Launch (1xH200 by default). Bind-mount the fork (live code), gaze data, rehearsal data.
#    GAZE_OBJECTIVE / GAZE_SPLIT_NAME / GAZE_SPECIALIZE_RATIO are read by our dataset + mixture.
# ----------------------------------------------------------------------------------------- #
# Multi-node needs the container on the HOST network so cross-node NCCL + the torchrun
# rendezvous can reach the head node's addr:port (the default bridge network isolates them).
NET_ARGS=()
[ "$NNODES" -gt 1 ] && NET_ARGS=( --network host )
# HF_CACHE is always writable now (offline mode stages to local scratch up-front).
# Offline just disables network at train time.
HF_OFFLINE_ENV=()
if [ "$HF_OFFLINE" -eq 1 ]; then
  HF_OFFLINE_ENV=( -e HF_HUB_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 )
fi
RUN=(
  docker run --rm --gpus all --shm-size=32g
  ${NET_ARGS[@]+"${NET_ARGS[@]}"}
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
  ${HF_OFFLINE_ENV[@]+"${HF_OFFLINE_ENV[@]}"}
  -e OMP_NUM_THREADS=8
  # Reduce CUDA allocator fragmentation. Matters when the gaze inference eval gathers a
  # full unsharded model onto one GPU and then frees it -- without this the freed blocks
  # are fragmented and training can OOM on resume at the larger seq lengths.
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  # Cross-node NCCL timeout (rendezvous + first all-gather can be slow on multi-node);
  # default 20 min, override via NCCL_TIMEOUT_MINUTES in the host env.
  -e NCCL_TIMEOUT_MINUTES="${NCCL_TIMEOUT_MINUTES:-20}"
  -e GAZE_DATA_DIR=/gaze-data
  -e GAZE_OBJECTIVE="$GAZE_OBJECTIVE"
  -e GAZE_SPLIT_NAME="$GAZE_SPLIT_NAME"
  -e GAZE_SPECIALIZE_RATIO="$SPECIALIZE_RATIO"
  -e WANDB_API_KEY="$WANDB_KEY"
  -e WANDB_PROJECT="$WANDB_PROJECT"
  -e WANDB_ENTITY="$WANDB_ENTITY"
  ${AWS_ENVS[@]+"${AWS_ENVS[@]}"}
)
# Forward NIC selection for NCCL/Gloo if the host pins it (some cloud nodes need an explicit
# interface, e.g. NCCL_SOCKET_IFNAME=eth0). Only passed through when set in the host env.
for v in NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME NCCL_IB_DISABLE; do
  [ -n "${!v:-}" ] && RUN+=( -e "$v=${!v}" )
done
# Point wandb at a self-hosted server when a base URL is given (required for "local-" keys).
[ -n "$WANDB_BASE_URL_" ] && RUN+=( -e WANDB_BASE_URL="$WANDB_BASE_URL_" )
# Pass the HF token under BOTH names the code reads (HF_ACCESS_TOKEN in tokenizer.py +
# most loaders; HF_TOKEN in academic_video_datasets.py), so gated repos authenticate either way.
[ -n "$HF_TOKEN" ] && RUN+=( -e HF_ACCESS_TOKEN="$HF_TOKEN" -e HF_TOKEN="$HF_TOKEN" )
# $TORCHRUN was composed above (shared with the --no-docker path).
RUN+=(
  "$IMAGE"
  bash -lc "set -e; pip install -e . >/dev/null 2>&1 || true; \
    $TORCHRUN launch_scripts/sft.py ${SFT_ARGS[*]}"
)

if [ "$DRY_RUN" -eq 1 ]; then
  echo ">> DRY RUN -- command that would execute:"
  printf '%q ' "${RUN[@]}"; echo
  exit 0
fi

exec "${RUN[@]}"
