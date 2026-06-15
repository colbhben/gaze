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

## 2. Gaze SFT smoke run (1×H200) — `training/gaze_sft.sh`

> **Full parameter reference:** every wrapper flag and every underlying `sft.py` / `TrainConfig`
> knob (incl. dotlist overrides) is documented in [`sft_params.md`](sft_params.md).

The gaze run is a **specialize-then-rehearse SFT**: train mostly on our gaze video-point data,
rehearsing a slice of the general Molmo2 SFT mixture to avoid forgetting. One script does the
whole chain (validate → stage data → author the 92/8 mix → compose flags → S3 checkpoint →
`torchrun`). It is the launcher to use for the gaze milestone:

```sh
training/gaze_sft.sh --name gaze-smoke-01 \
  --wandb-key "$WANDB_API_KEY" --wandb-project gaze --wandb-entity my-team \
  --gaze-data-dir /home/ubuntu/gaze-extract-full \
  --molmo-data-dir /home/ubuntu/molmo2-data \
  --gaze-objective first --max-duration 200
```

- **Training objective (flexible).** `--gaze-objective first` (default) trains *predict the FIRST
  gaze point (t0) given the full video*; `--gaze-objective all` trains *predict ALL per-frame gaze
  points*. The model always sees the full clip — only the target point set changes (sliced at
  dataset-construction time, not a loss mask). New objectives = a new entry in
  `olmo/data/gaze_datasets.py::_OBJECTIVE_KEEP`.
- **Annotation = INPUT, gaze = OUTPUT.** The clip's annotation text is fed *with the video* as the
  model's input prompt (the `video_gaze_point` style template, e.g. *"Given the activity …, point
  to where the camera wearer is looking."*); the trained OUTPUT is the gaze `<points>` string only.
- **Specialize mix.** `gaze_specialize` = `--specialize-ratio 0.92` → 92% gaze / 8% rehearse
  (Molmo2-recommended). The 8% rehearse blend is a compact slice of the general mixture
  (`tulu4`, `pixmo_ask_model_anything`, `chart_qa`, `pixmo_points`, a video MC set, hardcodes).
- **Metrics → wandb.** A held-out gaze **val** split is evaluated with `GazePointEval` →
  **L2 distance** (normalized 0–100 space) + **accuracy@radius** (5/10/15) + valid-prediction rate,
  logged to wandb alongside train loss. wandb credentials are **required** (`--wandb-key/-project/
  -entity` or `WANDB_API_KEY/WANDB_PROJECT/WANDB_ENTITY`); the script fails fast if any is missing.
- **Checkpoints → S3 (never /nfs).** Default `--save-folder
  s3://far-research-internal/colbhben/gaze/molmo/runs/<name>`; molmo2's checkpointer uploads
  natively when `save_folder` is a cloud URL. `/nfs` is a strictly **read-only** mount — the script
  refuses any `/nfs` save_folder or data dir and mounts data dirs `:ro`.
- **Data staging.** Pass already-local `--gaze-data-dir`/`--molmo-data-dir`, or `--stage-from
  <path-or-s3>` to copy the joint manifest + split pointers OUT of a read-only source into local
  scratch first. Required layout under `--gaze-data-dir`: `joint/manifest.jsonl` (self-contained,
  ABSOLUTE video paths — see `gaze curate join-manifests`) + `splits/<name>/{train,val}.jsonl`.
- **All training knobs exposed** with profile defaults (h200: `--seq-len 8192`, `--device-batch-size 4`,
  `--global-batch-size 32`), plus `--cp-degree 1`, `--llm-lr 1e-5`, `--vit-lr 5e-6`, `--connector-lr 5e-6`, `--warmup 200`,
  `--alpha-f 0.1`, `--save-interval`, `--global-batch-size`, plus `--max-duration` (short for smoke).
  Anything after `--` is passed verbatim as OmegaConf dotlist overrides.

Use `--dry-run` to print the exact `docker run … torchrun … sft.py …` command without launching
(handy to verify flags/staging off-GPU).

### Legacy NFS launcher — `training/molmo_train.sh`

The older `training/molmo_train.sh` writes checkpoints to `/nfs/.../runs/<name>` and predates the
S3/wandb/objective requirements. Kept for the non-gaze debug smoke (`--mixture debug --debug`);
prefer `gaze_sft.sh` for the gaze run.

`sft.py` positional args are `checkpoint` then `mixture`. Built-in mixtures now include
`gaze_specialize`/`gaze_rehearse` (ours) plus `debug`, `molmo2`, `tracking`, `pointing*`,
`vpointing*`. Long-context flags (mirror from molmo2 README when needed):
`--seq_len=36864 --model.mm_preprocessor.video.max_frames=384 --device_batch_size=1`.

### Mounts and outputs (`gaze_sft.sh`)
| Host | Container | Mode | Holds |
|---|---|---|---|
| `third_party/molmo2` | `/molmo2` | rw | our fork (working dir; `pip install -e .`) |
| `--gaze-data-dir` (LOCAL) | `/gaze-data` | **ro** | `joint/manifest.jsonl` + `splits/<name>/{train,val}.jsonl` |
| `--molmo-data-dir` (LOCAL) | `/data/molmo` | **ro** | `Molmo2-Data/` rehearsal + HF cache |

Checkpoints/runs go to the **S3** `--save-folder` (default
`s3://far-research-internal/colbhben/gaze/molmo/runs/<name>`). Nothing is written to `/nfs`
(read-only); data dirs are mounted `:ro`.

### Env vars (`gaze_sft.sh` sets these; secrets come from your shell/flags)
Container env: `OLMO_SHARED_FS=1`, `MOLMO_DATA_DIR=/data/molmo`, `HF_HOME=/data/molmo/huggingface`,
`OMP_NUM_THREADS=8`, `GAZE_DATA_DIR=/gaze-data`, `GAZE_OBJECTIVE`, `GAZE_SPLIT_NAME`,
`GAZE_SPECIALIZE_RATIO`, `WANDB_API_KEY/PROJECT/ENTITY` (**required**), AWS creds (passed through if
set, for the S3 upload). Optional: `HF_ACCESS_TOKEN`.

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

## Gaze data piping (how a clip becomes a training example)

`olmo/data/gaze_datasets.py::GazeVideoPoint` reads the split-pointer JSONL
(`GAZE_DATA_DIR/splits/<name>/<split>.jsonl`, `{id,dataset,video}`) and joins the joint manifest
(`GAZE_DATA_DIR/joint/manifest.jsonl`) by `id`, then builds the exact `Molmo2VideoPoint` example
schema. Load-bearing conversions (covered by `third_party/molmo2/tests/test_gaze_dataset.py`):
- **Coordinates pixel → 0–100.** Our manifest stores raw pixels on a `resolution`×`resolution`
  padded frame; Molmo2's point formatter divides by `scale=100` and clamps to `[0,1]`, so the
  dataset normalizes `pixel/resolution*100` (else every point clamps to the frame edge). The eval
  GT (`metadata.gt_abs_triplets`) keeps raw pixels — predictions are parsed back to pixels via the
  recorded frame dims.
- **Real gaze only.** Frames with no/invalid gaze (empty point list, `frame_mask==0`) are dropped
  *before* objective slicing, so `first` never picks an empty t0 and the target is always a real
  point (not "There are none.").
- **Objective slicing.** `first` keeps the earliest real-gaze frame (1 point); `all` keeps every
  real-gaze frame.

The registrations: `gaze_video_point` (train) + `gaze_video_point_eval` (val) in
`olmo/data/get_dataset.py`; the `video_gaze_point` style + prompt templates in
`olmo/preprocessing/data_formatter.py`; the `gaze_specialize`/`gaze_rehearse` mixtures in
`launch_scripts/sft.py`; the `GazePointEval` metric in `olmo/eval/evaluators.py` (wired via
`gaze_point_eval` in `inf_evaluator.py` + `eval_utils.py`).

## Caveats — recap

These are detailed above; the load-bearing ones for the gaze run:
- **GPU.** 1×**H200 (sm_90)** — the prebuilt image's flash-attn (`90;100`) works; default attention
  is `sdpa`. (L40S support dropped for this milestone.)
- **Checkpoint format.** Start from an **olmo-native 4B** checkpoint (`config.yaml` +
  `model_and_optim/`); `sft.py` does not accept HF safetensors. `--checkpoint /data/molmo/Molmo2-4B-SFT`
  by default — confirm the staged 4B is olmo-native (`gaze-84q`).

## Verification ladder
1. `git submodule update --init` → `third_party/molmo2/launch_scripts/sft.py` present.
2. **Offline unit tests** (no GPU/torch): from `third_party/molmo2`,
   `python3 -m unittest tests.test_gaze_dataset tests.test_gaze_point_eval` — dataset transform
   (coords/objective/real-gaze/eval-GT) + the L2/acc@radius scoring kernel.
3. `docker pull ghcr.io/allenai/molmo2:latest` succeeds (or fork Dockerfile builds).
4. **Dry-run** the launcher off-GPU: `training/gaze_sft.sh --name dbg --wandb-* … --gaze-data-dir …
   --molmo-data-dir … --dry-run` prints the exact `docker/torchrun/sft.py` command.
5. **Dataloader smoke** (H200, sdpa): `get_dataset_by_name("gaze_video_point")` loads + a few
   batches iterate through the Molmo2 preprocessor (closes `gaze-h7k` step 5).
6. **Specialize-then-rehearse smoke** (1×H200): `training/gaze_sft.sh --name gaze-smoke-01
   --gaze-objective first --max-duration 200 …` runs ~200 steps, checkpoints to S3, logs train loss
   + gaze L2/acc to wandb. Then flip `--gaze-objective all` to verify the objective toggle
   (`gaze-8yc.12`).
