# Molmo2 training on L40 / L40S (Ada, sm_89)

`Dockerfile.l40s` is a patched copy of `third_party/molmo2/Dockerfile` that makes the
Molmo2 training image run on **L40 / L40S** GPUs. It is the artifact mirrored to
`s3://far-research-internal/colbhben/gaze/molmo/`.

## Why a patch is needed

The stock image (`docker pull ghcr.io/allenai/molmo2:latest`) compiles **flash-attn 2.8.3
for `sm_90` and `sm_100` only**:

```dockerfile
RUN FLASH_ATTN_CUDA_ARCHS="90;100" pip install ... flash-attn==2.8.3
```

| Arch | GPU family | Stock image |
|------|------------|-------------|
| `sm_90`  | Hopper — **H100 / H200** | ✅ supported |
| `sm_100` | Blackwell — **B200**     | ✅ supported |
| `sm_89`  | Ada — **L40 / L40S**, L4, RTX 4090 | ❌ **missing** |

On an L40S the stock kernels fail at the first attention call
(`olmo/nn/flash_attention_api.py` → `flash_attn_func` / `flash_attn_varlen_func`) with:

```
RuntimeError: no kernel image is available for execution on the device
```

## What the patch changes

Exactly two lines, both arch lists, both overridable via `--build-arg`:

| Component | Upstream | Patched |
|-----------|----------|---------|
| flash-attn | `FLASH_ATTN_CUDA_ARCHS="90;100"` | `"89;90;100"` |
| grouped_gemm | `TORCH_CUDA_ARCH_LIST="9.0 10.0"` | `"8.9 9.0 10.0"` |

H200 and B200 stay supported (their archs are retained). Everything else — torch
2.9.1/cu128, bf16 AMP, vLLM, ring-flash-attn, the dense Qwen3 4B/8B path — is byte-for-byte
upstream and was already Ada-compatible.

## Building

The CUDA `*-devel` base image is x86_64 only, and flash-attn must compile against the
target arch, so **build on an x86_64 host with the NVIDIA Container Toolkit** (e.g. the
L40S box itself or any CUDA build host). Cross-building from an arm64 Mac is not supported.

```bash
# from repo root
docker build -f training/docker/Dockerfile.l40s -t molmo2:l40s third_party/molmo2

# then point the launcher at the local image
training/gaze_sft.sh --name gaze-l40s-01 --image molmo2:l40s \
    --wandb-key "$WANDB_API_KEY" --wandb-project gaze --wandb-entity <team> \
    --gaze-data-dir /home/ubuntu/gaze-stage/gaze-data --molmo-data-dir /data/molmo
```

> Building flash-attn for an extra arch adds compile time (tens of minutes). The wheel is
> the dominant cost; the rest of the image is unchanged from upstream.

## Other L40S de-risk findings (`training/gaze_sft.sh`)

flash-attn is the only **hard build blocker**. The remaining items are runtime/capacity
concerns, ordered by likelihood of biting:

1. **VRAM is the real risk, not arch.** L40S has **48 GB**; the script's H200 (141 GB)
   defaults will OOM on a single L40S:
   - `--seq-len 16384`, `--device-batch-size 2`, and the 4B (let alone 8B) checkpoint.
   - Mitigations (all already exposed as flags / dotlist overrides):
     `--seq-len 4096..8192`, `--device-batch-size 1`, lower `--global-batch-size`,
     activation checkpointing, and prefer the **4B** checkpoint over 8B for a
     single-card smoke. Multi-card: raise `--cp-degree` (context parallel) and/or `--gpus`.

2. **`docker` is hard-required by the launcher.** `gaze_sft.sh` does
   `command -v docker >/dev/null || die` and invokes `docker run --gpus all`. The L40S
   host must have Docker + NVIDIA Container Toolkit. (This Mac has only `finch`, which is
   why the image could not be pulled locally — pull/build on the L40S host instead.)

3. **bf16 is fine; fp8 is not used.** Default precision is `amp_bf16` (Ada supports bf16).
   Float8 is explicitly `NotImplementedError` in `olmo/dist_util.py`, so there is no
   Hopper/Blackwell-only quant path to trip over.

4. **MoE / grouped_gemm is not on the gaze path.** Qwen3 4B (`qwen3_4b_instruct`) and 8B
   (`qwen3_8b`) are **dense** configs (`olmo/model_configs.py`), so grouped_gemm is never
   called. Patched anyway for completeness.

5. **No NVLink / lower interconnect.** L40S is PCIe-only (no NVLink). Single-card smoke is
   unaffected; multi-card runs will see slower all-reduce / CP communication than H200.
   The DOCA/OFED RDMA bits in the image are inert without the matching NICs — harmless.

6. **`torch.compile` is on by default** (`compile=CompilerConfig(mode="default")` in
   `sft.py`). It works on Ada but first-step compilation is slow; pass
   `compile=null` (dotlist) to disable if it causes issues during a quick smoke.
