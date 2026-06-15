# Molmo2 gaze SFT — parameter reference

Complete reference for every tunable parameter of the gaze SFT training harness. Two layers:

1. **`training/gaze_sft.sh`** — our wrapper. Exposes the ~34 flags you'll touch day-to-day, each with
   a Molmo2-recommended default, and composes them into the call below.
2. **`launch_scripts/sft.py`** (`third_party/molmo2`) — the trainer entrypoint. Takes a few argparse
   flags plus **arbitrary `key=value` OmegaConf dotlist overrides** that reach into the full
   `TrainConfig` (and its nested configs). Anything the wrapper doesn't expose is still reachable.

> Canonical sources (edit these, then update this doc):
> - Wrapper flags + defaults: `training/gaze_sft.sh:33-81` (usage), `:95-145` (default vars), `:348-372` (compose).
> - sft.py argparse + the constructed `TrainConfig`: `third_party/molmo2/launch_scripts/sft.py:557-568`, `:673-758`.
> - `TrainConfig`: `third_party/molmo2/olmo/train/trainer_config.py:511-876`.
> - `OptimizerConfig` / `SchedulerConfig`: `third_party/molmo2/olmo/train/optim.py:49-93`, scheduler block.
> - `DataLoaderConfig` / `PackingConfig`: `third_party/molmo2/olmo/data/data_loader.py:86-150`, `olmo/data/dynamic_packer.py:303-319`.
> - `ParallelismConfig` / context-parallel: `trainer_config.py:250-309,452-502`. See also `third_party/molmo2/docs/context_parallel.md`.

Companion docs: [`molmo2_training.md`](molmo2_training.md) (workflow/concepts), bead `gaze-67t.2` (settings math).

---

## How invocation works

```
training/gaze_sft.sh <wrapper-flags> [-- <raw sft.py dotlist overrides>]
        │
        └─> torchrun --nproc-per-node=$GPUS launch_scripts/sft.py \
                <checkpoint> <mixture> \
                --seq_len --device_batch_size --cp_degree --num_workers \   # argparse
                save_folder=... optimizer.llm_learning_rate=... ...          # dotlist -> TrainConfig
                <your -- extras>                                             # merged last, win ties
```

- The wrapper translates its friendly flags into the positional args, the four sft.py argparse flags,
  and a set of dotlist overrides (`gaze_sft.sh:348-372`).
- Everything after a literal `--` is forwarded verbatim as additional dotlist overrides, merged **last**
  so it overrides both the wrapper and the TrainConfig defaults (`sft.py:758`). This is how you reach
  the ~100+ advanced knobs the wrapper doesn't surface.
- Dotlist syntax is `dotted.path=value`, e.g. `optimizer.llm_weight_decay=0.01`, `compile=null`,
  `data.packing.buffer_size=64`. Values parse as YAML (so `null`, `true`, numbers, lists work).

---

## 1. Wrapper flags (`gaze_sft.sh`)

`[req]` = required. Every default is overridable.

### Run / credentials
| Flag | Default | Meaning |
|---|---|---|
| `--name NAME` | — `[req]` | Run name; also fills the default S3 `save_folder`. → `run_name`. |
| `--wandb-key KEY` | `$WANDB_API_KEY` `[req]` | W&B API key (no key is hardcoded; fails fast if missing). |
| `--wandb-project PROJ` | `colbhben-gaze` | W&B project. → `WANDB_PROJECT` env → `wandb.project`. |
| `--wandb-entity ENT` | `far-wandb` | W&B entity/team. → `wandb.entity`. |
| `--wandb-base-url URL` | `https://far.wandb.io` | Self-hosted W&B server (required for `local-` keys). |
| `--hf-token TOK` | `$HF_ACCESS_TOKEN` | HuggingFace token (gated datasets/checkpoints). |

### Data
> No path may live on `/nfs` at run time — `/nfs` is a read-only mount; stage to local scratch first.

| Flag | Default | Meaning |
|---|---|---|
| `--gaze-data-dir DIR` | — `[req]` | Local dir with `joint/manifest.jsonl` + `splits/<name>/{train,val}.jsonl`. |
| `--gaze-split-name NAME` | `v1_95_05` | Split subdir under `splits/`. |
| `--molmo-data-dir DIR` | — `[req]` | Local Molmo2-Data rehearsal root (the 8% slice draws from here). |
| `--stage-from SRC` | — | Copy data out of an `/nfs` path or `s3://` URL into local scratch first. |
| `--local-scratch DIR` | `/home/ubuntu/gaze-stage` | Where `--stage-from` lands. |
| `--hf-cache DIR` | `<scratch>/hf-cache` | Writable HF cache (never `/nfs`). |

### Mixture / objective
| Flag | Default | Meaning |
|---|---|---|
| `--mixture NAME` | `gaze_specialize` | Mixture name resolved in `sft.py::get_training_mixture`. |
| `--gaze-objective first\|all` | `first` | `first` = predict t0 gaze point; `all` = per-frame points. Target slicing at dataset-construction time (not a loss mask). |
| `--specialize-ratio R` | `0.92` | Gaze fraction; `1-R` is the rehearse slice. `R=1.0` drops rehearse groups (gaze-only; `sft.py:542-544`). |

### Hardware profile
Selects the memory-bound defaults (image, seq_len, device/global batch). Any explicit flag overrides.

| `--profile` | GPUs | image | seq_len | device_batch | global_batch |
|---|---|---|---|---|---|
| `l40` (default) | 8×L40 48 GB | `molmo2:l40s` | 8192 | 1 | 64 |
| `h200` | H200 141 GB | `ghcr.io/allenai/molmo2:latest` | 8192 | 4 | 32 |

### Model / image
| Flag | Default | Meaning |
|---|---|---|
| `--checkpoint PATH` | `/data/molmo/Molmo2-4B-SFT` | olmo-native starting checkpoint (local path `s3://`, or in-container path). → positional + `initial_model_checkpoint`. |
| `--image IMG` | per `--profile` | Docker image. |

### Checkpointing
| Flag | Default | Meaning |
|---|---|---|
| `--save-folder URL` | `s3://far-research-internal/colbhben/gaze/molmo/runs/<name>` | Output dir, `s3://` or local; `/nfs` rejected. Native cloud upload. → `save_folder`. |
| `--save-interval N` | `2000` | Steps between checkpoint saves. → `save_interval`. |
| `--max-duration N` | sft.py default | **Total training steps** (e.g. `200` for a smoke). → `max_duration`. |

### Parallelism / batch
> Activation memory is bound by `seq_len × device_batch_size` per GPU and is **not** reduced by adding
> GPUs. For sequences longer than the profile default, prefer `--cp-degree` over a bigger `seq_len`.

| Flag | Default | Meaning |
|---|---|---|
| `--gpus N` | per profile (8) | `torchrun --nproc-per-node`. |
| `--cp-degree N` | `1` | Context-parallel degree; shards the sequence across N ranks. Must divide `--gpus`. `>1` auto-sets `compile=null`. → `--cp_degree`. |
| `--seq-len N` | per profile | Packed sequence length (tokens). → `--seq_len` + `data.sequence_length`. |
| `--device-batch-size N` | per profile | Per-rank microbatch. The primary HBM-filling knob. → `--device_batch_size`. |
| `--global-batch-size N` | per profile | Effective global batch; must divide DP degree (`gpus/cp_degree`). → `global_train_batch_size`. |
| `--num-workers N` | `6` | Dataloader workers. → `--num_workers`. |

### Optimizer / schedule
| Flag | Default | Meaning |
|---|---|---|
| `--llm-lr LR` | `1e-5` | LLM learning rate. → `optimizer.llm_learning_rate`. |
| `--vit-lr LR` | `5e-6` | ViT learning rate. → `optimizer.vit_learning_rate`. |
| `--connector-lr LR` | `5e-6` | Connector learning rate. → `optimizer.connector_learning_rate`. |
| `--warmup N` | `200` | Warmup steps (applied to llm/vit/connector). → `scheduler.*_t_warmup`. |
| `--alpha-f F` | `0.1` | Scheduler final-LR fraction (10% floor). → `scheduler.alpha_f`. |

### Utility
| Flag | Meaning |
|---|---|
| `--dry-run` | Print the docker run / torchrun command without executing. |
| `-h`, `--help` | Show usage. |
| `-- <extra...>` | Everything after `--` is forwarded as raw sft.py dotlist overrides (merged last). |

---

## 2. sft.py argparse flags (set by the wrapper; override directly only when calling sft.py raw)

| Flag | Default | Meaning |
|---|---|---|
| `checkpoint` (positional) | — | olmo-native starting checkpoint. |
| `mixture` (positional) | `0.0.1` | Mixture name (`gaze_*`, `pointing*`, `vpointing*`, `image-only-v5*`, `debug`, …). Drives which eval tasks run (`sft.py:571-600`). |
| `--debug` | off | Tiny model + small dataset for a smoke; disables wandb, ft_llm/ft_vit. |
| `--model` | `video` | Model family selector passed to `get_model`. |
| `--seq_len` | `16384` | Sequence length (wrapper sets from `--seq-len`). |
| `--device_batch_size` | `2` | Per-rank microbatch (wrapper sets from `--device-batch-size`). |
| `--max_loss_examples` | `2048` | Cap on loss-eval examples. |
| `--max_inf_eval_examples` | `1280` | Cap on inference-eval examples. |
| `--prefetch_factor` | `4` | Dataloader prefetch factor. |
| `--num_workers` | `6` | Dataloader workers. |
| `--cp_degree` | `1` | Context-parallel degree. |

> **Gaze-aware defaults baked into sft.py** (`:626-672`): for `gaze*` mixtures it sets the eval task to
> `gaze_video_point_eval`, logs 8 prediction cards, `log_interval=1`, and `inf_eval_interval=4`.
> Override any of these with `-- inf_eval_interval=<n>` etc.

---

## 3. Advanced overrides via dotlist (not exposed by the wrapper)

These are the most useful `TrainConfig` / nested fields you might set with `-- key=value`. Defaults
shown are the **TrainConfig** defaults; note sft.py overrides some at construction (`:673-753`) —
the "sft.py sets" column reflects what actually runs for a gaze job.

### TrainConfig — fine-tuning scope & loss (`trainer_config.py`)
| Dotlist key | TrainConfig default | sft.py sets | Meaning |
|---|---|---|---|
| `ft_llm` | `true` | `not --debug` | Tune LLM params. |
| `ft_vit` | `true` | `not --debug` | Tune ViT params. |
| `ft_connector` | `true` | `true` | Tune V/L connector. |
| `ft_embedding` | `lm_head` | — | Which embeddings to tune. |
| `max_grad_norm` | `null` | `1` | Gradient-norm clip. |
| `max_grad_norm_ratio` | `null` | — | Adaptive clip vs running grad-norm avg (overrides `max_grad_norm`). |
| `batch_divisor` | `global_batch` | `global_batch` | Loss normalization in distributed settings. |
| `softmax_auxiliary_loss` | `false` | `true` | z-loss (softmax-normalizer regularizer). |
| `softmax_auxiliary_loss_scale` | `1e-4` | `1e-4` | z-loss scale. |
| `response_logits_only` | `true` | `true` | Compute logits only for non-zero-weight tokens. |
| `precision` | `null` | `amp_bf16` | `amp_bf16` / `amp_fp16` / `fp32`. |
| `activation_checkpointing` | `true` | — | Trade compute for activation memory. |
| `compile` | `null` | `CompilerConfig(mode=default)` | torch.compile; **must be `null` when cp_degree>1**. |
| `compile_loss` | `false` | `true` | Compile the loss fn. |
| `fused_loss` | `null` | `false` | flash-attn fused CE loss. |
| `seed` | `6198` | `6198` | Global RNG seed. |

### TrainConfig — duration, checkpointing, resume
| Dotlist key | Default | sft.py sets | Meaning |
|---|---|---|---|
| `max_duration` | `10000` | `300000` | Train length; bare int = steps, or `"2e12T"` = tokens. (Wrapper `--max-duration`.) |
| `stop_at` / `stop_after` | `null` | `stop_at=${max_duration}` | Hard stop at / after a step count. |
| `global_train_batch_size` | `512` | `128` | Effective global batch. (Wrapper `--global-batch-size`.) |
| `device_train_microbatch_size` | `16` | `--device_batch_size` | Per-device microbatch. |
| `save_interval` | `1000` | `2000` | Steps between sharded saves. |
| `save_interval_ephemeral` | `null` | — | Frequent throwaway restart checkpoints (keeps only latest). |
| `save_num_checkpoints_to_keep` | `-1` | `1` | How many sharded checkpoints to retain (`-1`=all). |
| `save_overwrite` | `false` | `true` | Overwrite existing files in `save_folder`. |
| `save_final_optim` | `true` | `false` | Save final optimizer state. |
| `save_final_unsharded_checkpoint` | `false` | `false` | Emit an unsharded checkpoint at the end. |
| `save_at` | `null` | — | Also save at a specific step. |
| `load_path` | `null` | `null` | Resume from a sharded/unsharded checkpoint (overrides `initial_model_checkpoint`). |
| `initial_model_checkpoint` | `null` | `<checkpoint>` | Init model weights only (no trainer state). |
| `allow_resume` | `false` | `true` | Auto-resume if a checkpoint exists in `save_folder`. |
| `restore_dataloader` | `true` | — | Restore dataloader position on resume (set `false` to retrain on new data). |
| `reset_optimizer_state` / `reset_trainer_state` | `false` | — | Drop optim / trainer state from `load_path`. |
| `fast_forward_batches` | `null` | — | Skip N batches on resume. |
| `time_limit` | `null` | `null` | Wall-clock limit; saves + exits early. |
| `extra_steps_after_cancel` | `0` | — | Train a few extra steps post-cancel for metric overlap. |

### TrainConfig — evaluation & logging
| Dotlist key | Default | sft.py sets (gaze) | Meaning |
|---|---|---|---|
| `eval_interval` | `1000` | `-1` | Steps between **loss** evals (`-1`=never). |
| `inf_eval_interval` | `-1` | `4` | Steps between **inference** evals (gaze L2 / acc@radius). |
| `eval_on_last_step` | `true` | `true` | Always eval on the final step. |
| `eval_on_load` | `false` | — | Run evals immediately on resume. |
| `eval_on` | `()` | — | Extra explicit step numbers to eval at. |
| `console_log_interval` | `1` | `1`/`20` | Console log cadence. |
| `save_inloop_predictions` | `true` | — | Persist in-loop eval predictions into the checkpoint dir. |
| `wandb.log_interval` | — | `1`/`20` | W&B log cadence (1 for gaze/debug). |
| `gen1_gc_interval` | `1` | — | GC cadence (`null` = automatic). |
| `python_profiling` / `torch_profiling` | `false` | — | Profile batches 6–8. |

### OptimizerConfig (`optim.py:49-93`) — prefix `optimizer.`
| Dotlist key | Default | sft.py sets | Meaning |
|---|---|---|---|
| `optimizer.name` | `lionw` | `adamw` | Optimizer type. |
| `optimizer.llm_learning_rate` | `1e-4` | `1e-5` | LLM LR (wrapper `--llm-lr`). |
| `optimizer.vit_learning_rate` | `1e-4` | `5e-6` | ViT LR (wrapper `--vit-lr`). |
| `optimizer.connector_learning_rate` | `1e-4` | `5e-6` | Connector LR (wrapper `--connector-lr`). |
| `optimizer.frame_selector_learning_rate` | `1e-4` | `1e-4` | Frame-selector LR. |
| `optimizer.{llm,vit,connector,frame_selector}_weight_decay` | `0.0` | — | Per-module weight decay. |
| `optimizer.{llm,vit,connector,frame_selector}_betas` | `(0.9,0.95)` | — | Per-module Adam betas. |
| `optimizer.{llm,vit,connector,frame_selector}_eps` | `1e-6` | — | Per-module Adam eps. |
| `optimizer.metrics_log_interval` | `-1` | — | Optimizer-metrics log cadence. |

### SchedulerConfig (`optim.py`) — prefix `scheduler.`
| Dotlist key | Default | sft.py sets | Meaning |
|---|---|---|---|
| `scheduler.name` | `cosine_with_warmup` | `multimodal` | Scheduler type. |
| `scheduler.units` | `steps` | — | Warmup/`t_max` units. |
| `scheduler.{llm,vit,connector,frame_selector}_t_warmup` | `200` | `200` | Per-module warmup (wrapper `--warmup`). |
| `scheduler.alpha_f` | `0.1` | `0.1` | Final-LR fraction (wrapper `--alpha-f`). |
| `scheduler.t_max` | `null` | — | Decay horizon (defaults to `max_duration`). |
| `scheduler.warmup_min_lr` | `null` | `0.0` | LR at step 0. |
| `scheduler.grad_clip_warmup_steps` / `_factor` | `null` | — | Looser grad clip during warmup. |

### DataLoaderConfig (`data_loader.py:86-150`) — prefix `data.`
| Dotlist key | Default | sft.py sets | Meaning |
|---|---|---|---|
| `data.sequence_length` | `null` | `seq_len` | Packed sequence length (mirrors `--seq_len`). |
| `data.max_text_seq_len` | `null` | `null` | If set, seq_len = this + max MM tokens. |
| `data.pad` | `to_max` | `to_max` | Padding mode. |
| `data.shuffle` | `true` | `true` | Shuffle training data. |
| `data.drop_last` | `false` | `true` | Drop the final ragged batch. |
| `data.num_workers` | `0` | `--num_workers` | Dataloader workers. |
| `data.prefetch_factor` | `null` | `--prefetch_factor` | Prefetch per worker. |
| `data.pin_memory` | `true` | `true` | Pin host memory. |
| `data.persistent_workers` | `false` | — | Keep workers alive across epochs. |
| `data.seed` | MISSING | `50189` | Dataloader RNG seed. |
| `data.enable_variable_sized_token_pooling` | `true` | — | Variable-size vision-token pooling. |

### PackingConfig (`dynamic_packer.py:303-319`) — prefix `data.packing.`
| Dotlist key | Default | sft.py sets | Meaning |
|---|---|---|---|
| `data.packing.buffer_size` | `32` | `48` | Examples buffered for the packing solver (higher = tighter packing, more host RAM). |
| `data.packing.mode` | `dynamic_solver` | — | Packing algorithm. |
| `data.packing.text_weight` | `1.0` | — | Solver cost weight for text tokens. |
| `data.packing.image_weight` | `1.0` | `30` | Solver cost weight for image tokens. |
| `data.packing.shortcut_max_len_images` | `false` | `false` | Shortcut for max-length image examples. |
| `data.packing.cp_world_size` | `1` | `--cp_degree` | CP group size for image-shard boundary precompute. |

### ParallelismConfig (`trainer_config.py:452-502`) — prefix `parallelism.`
| Dotlist key | Default | sft.py sets | Meaning |
|---|---|---|---|
| `parallelism.context_parallel_config.degree` | `1` | `--cp_degree` | Context-parallel degree (sequence sharding). See `docs/context_parallel.md`. |
| `parallelism.tensor_parallel_config.degree` | `1` | — | Tensor-parallel degree. |
| `parallelism.data_parallel_replicate_degree` | `1` | — | DP replicate (HSDP outer). |
| `parallelism.data_parallel_shard_degree` | `-1` | — | DP shard (`-1` = fill remaining ranks). |
| `parallelism.context_parallel_rotate_method` | `allgather` | — | CP K/V rotation method. |

### FSDPConfig — prefix `fsdp.`
| Dotlist key | Default | sft.py sets | Meaning |
|---|---|---|---|
| `fsdp.fsdp2` | — | `true` | Use FSDP2. |
| `fsdp.param_dtype` / `fsdp.reduce_dtype` | per config | — | Param / gradient-reduce dtypes. |

---

## 4. Common recipes

```sh
# Smoke (200 steps, single H200)
training/gaze_sft.sh --name gaze-smoke --profile h200 --gpus 1 \
  --gaze-data-dir /home/ubuntu/gaze-extract --molmo-data-dir /home/ubuntu/molmo2-data \
  --max-duration 200

# Long sequence that won't fit one GPU: shard with CP (cp must divide gpus; compile auto-disabled)
training/gaze_sft.sh --name gaze-cp --profile h200 --gpus 8 --cp-degree 4 --seq-len 36864 \
  --gaze-data-dir ... --molmo-data-dir ...

# Reach an unexposed knob via dotlist (weight decay + tighter packing + disable inf eval)
training/gaze_sft.sh --name gaze-x --gaze-data-dir ... --molmo-data-dir ... \
  -- optimizer.llm_weight_decay=0.01 data.packing.buffer_size=64 inf_eval_interval=-1
```

> Sanity rules enforced by the wrapper (`gaze_sft.sh:250-255`): `gpus % cp_degree == 0`, and
> `global_batch % (gpus/cp_degree) == 0`. `save_folder` may not be on `/nfs`.
