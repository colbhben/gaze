# Molmo2 VLM gaze training

This repo curates gaze training data; the **trainer** is the `allenai/molmo2` package, vendored as a
git submodule (our fork) at `third_party/molmo2`. Training is **Docker-first** — we run the prebuilt
AllenAI image rather than building molmo2's heavy CUDA deps (torch+cu128, flash-attn, torchcodec,
grouped_gemm) into the data-extraction venv.

- **Backbone:** a released **Molmo2 VLM** checkpoint. Inherits the native `<tracks>`/video-point
  head, so gaze drops in as a video-point task with no new modality. (We train the VLM output only;
  no robot-action/VLA components.)
- **Data env vs training env:** data extraction uses the one-step venv (`./scripts/setup.sh`, no
  torch). Training uses Docker. Keep them separate.

## 1. Get the trainer

```sh
git submodule update --init --recursive          # checks out third_party/molmo2 (our fork)
```

The fork (`github.com/colbhben/molmo2`) is where our gaze dataset registration + data-mix config land
(bead `gaze-kp2`). Track our `main`; periodically rebase on upstream `allenai/molmo2`.

## 2. Run training

```sh
# Trainer debug smoke (see caveats — needs H200/B200 and an olmo-native checkpoint):
training/molmo_train.sh --gpus 1 --mixture debug --debug --name dbg

# Real specialize-then-rehearse, once gaze-kp2 registers the mixture:
training/molmo_train.sh --gpus 8 --mixture gaze_rehearse --name gaze-smoke-01
```

`training/molmo_train.sh` pulls `ghcr.io/allenai/molmo2:latest` (falls back to building the fork's
`Dockerfile` if the pull fails), bind-mounts `third_party/molmo2 → /molmo2` (our fork's code runs
live) and `/nfs/colbhben/gaze/molmo → /data/molmo`, exports the molmo2 env vars, and invokes:

```
torchrun --nproc-per-node=<N> launch_scripts/sft.py <checkpoint> <mixture> \
  --save_folder=/data/molmo/runs/<name> --save_overwrite
```

`sft.py` positional args are `checkpoint` then `mixture`. Built-in mixtures: `debug`, `molmo2`,
`tracking`, `pointing*`, `vpointing*`. Long-context flags (mirror from molmo2 README when needed):
`--seq_len=36864 --model.mm_preprocessor.video.max_frames=384 --device_batch_size=1`.

### Mounts and outputs
| Host | Container | Holds |
|---|---|---|
| `third_party/molmo2` | `/molmo2` | our fork (working dir; `pip install -e .`) |
| `/nfs/colbhben/gaze/molmo` | `/data/molmo` | `Molmo2/` checkpoint, `Molmo2-Data/` rehearsal, `runs/` outputs |

Checkpoints/runs are written to `/nfs/colbhben/gaze/molmo/runs/<name>`.

### Env vars (set by the launcher; pass secrets via your shell)
`HF_DATASETS_OFFLINE=1`, `OLMO_SHARED_FS=1`, `MOLMO_DATA_DIR=/data/molmo`,
`HF_HOME=/data/molmo/huggingface`, `OMP_NUM_THREADS=8`. Optional: `WANDB_API_KEY`, `HF_ACCESS_TOKEN`.

## Caveats (read before the trainer smoke)

**GPU architecture.** Final training HW is **H200 (sm_90) / B200 (sm_100)**; the prebuilt image's
flash-attn is built for `sm_90;sm_100`. It will **not** run full GPU training on the L40S dev box
(`sumedhso-L40S`, sm_89). For A100 (sm_80) or an L40S smoke:
- run with `attn_implementation=sdpa` (Molmo2's adapter already defaults to sdpa) for a
  **dataloader/forward smoke only**, or
- rebuild flash-attn with the target arch added to `FLASH_ATTN_CUDA_ARCHS` (e.g. `80;89;90;100`).

**Checkpoint format.** `launch_scripts/sft.py` loads an **olmo-native** checkpoint (`config.yaml` +
`model.pt` / `model_and_optim/`); it does not accept HF `config.json`+safetensors and there is no
reverse converter. We therefore initialize from a **released Molmo2 VLM** checkpoint in olmo-native
form — staged directly if AllenAI ships one, else assembled via
`third_party/molmo2/scripts/prepare_pretrained_model.py` (builds olmo-native from the released
vision+LLM backbones). Staged to `/nfs/colbhben/gaze/molmo/Molmo2-VLM-olmo/`. Tracked in `gaze-84q`.

**ffmpeg on the dev box.** The data pipeline needs system ffmpeg; `sumedhso-L40S` is missing it —
`sudo apt-get install -y ffmpeg` (or `conda install -c conda-forge ffmpeg`). Independent of training.

## Verification ladder
1. `git submodule update --init` → `third_party/molmo2/launch_scripts/sft.py` present.
2. `docker pull ghcr.io/allenai/molmo2:latest` succeeds (or fork Dockerfile builds).
3. **Dataloader smoke** (L40S OK, sdpa): construct `Molmo2VideoTrack`, iterate staged gaze rows —
   this is `gaze-h7k` step 5 (fps=6 / pixel coords / frame_mask through the real loader).
4. **Trainer smoke** (H200/B200): `training/molmo_train.sh --gpus 1 --mixture debug --debug` runs a
   few steps and writes a checkpoint (needs the converted checkpoint above).
5. **Specialize-then-rehearse smoke:** `gaze-8yc.12`, once `gaze-kp2` registers `gaze_rehearse`.
