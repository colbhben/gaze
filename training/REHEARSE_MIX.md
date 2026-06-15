# Gaze specialize-then-rehearse SFT mixture

This documents the rehearsal data mixture used by the `gaze_specialize` (alias
`gaze_rehearse`) training mixture in
`third_party/molmo2/launch_scripts/sft.py::get_training_mixture`.

## What it is

We fine-tune the released **Molmo2 4B VLM** on our gaze video-point task using a
*specialize-then-rehearse* blend: most of the batch is the new gaze task, with a
small slice of the general Molmo2 SFT mixture mixed back in to prevent catastrophic
forgetting.

```
batch = gaze_w · gaze  +  (1 - gaze_w) · rehearse
```

`gaze_w` is `GAZE_SPECIALIZE_RATIO` (default **0.92**, i.e. 92% gaze / 8% rehearse;
the Molmo2-recommended specialize blend). With `GAZE_SPECIALIZE_RATIO=1.0` the
rehearse groups are dropped entirely (gaze-only training) so the gaze path can be
validated before the rehearsal data is staged.

## Rehearse group structure (matches the Molmo2 paper)

The rehearse slice is organized into the **paper's SFT data groups**, so it
preserves the model's general abilities in the paper's proportions. The paper's
seven groups and rates:

| Group            | Paper rate | In our mix? |
|------------------|-----------:|-------------|
| Captions/Long QA |      13.6% | ✅ |
| Image QA         |      22.7% | ✅ |
| Image Pointing   |       9.1% | ✅ |
| NLP (Tulu)       |       9.1% | ✅ |
| Video QA         |      18.2% | ❌ dropped (videos unstaged — see below) |
| Video Pointing   |      13.6% | ❌ dropped (videos unstaged — see below) |
| Video Tracking   |      13.6% | ❌ dropped (read-only cache — see below) |

> **All three video groups are currently dropped** because the Molmo2 video mp4s are
> still being extracted into this tree. The rehearse slice is therefore **image + text
> only** at the moment. This is a staging limitation, not a design choice — restore the
> video groups (the code for them is in git history / the restore checklist below) once
> their mp4s land. The 92% gaze group is itself video, so the model still trains on video
> every step.

Each group is a top-level entry in the training mixture; the trainer samples
**within** a group by per-dataset `sampling_rate` and **across** groups by the group
rate. Group rates are the paper rates **renormalized over the surviving groups** so
they sum to 1.0, then scaled by `(1 - gaze_w)`.

### Why Video Tracking is dropped (read-only cache)

`Molmo2-Data` is bind-mounted **read-only** into the training container (it lives on
a shared `/nfs` mount we must never write). Every tracking loader
(`mevis_*`, `lv_vis_*`, `revos_*`, `moca_*`, `ref_davis17_*`, `molmo2_video_track*`
in track/ground/single-point variants) materializes a *processed cache inside*
`MOLMO_DATA_DIR` at load time, which fails with `OSError: [Errno 30] Read-only file
system`. Until that cache is pre-built into the staged tree (or the loaders are
taught to cache elsewhere), tracking can't be rehearsed.

### Why Video QA and Video Pointing are dropped (videos unstaged)

These groups' datasets *construct* fine (their `.load()` reads JSON metadata) but their
examples reference video mp4s that **are not staged yet** — the extraction is still
running. A per-example **traversal probe** (open N example videos, not just `.load()`)
confirmed the videos are almost entirely absent:

| Dataset | example videos present |
|---------|------------------------|
| `llava_video_mc_academic` | 0 / 40 |
| `llava_video_oe_academic` | 0 / 40 |
| `clevrer` | 0 / 40 |
| `motionbench_train` | 6 / 40 |
| `academic_points_clip_63s_2fps` (video pointing) | 0 / 40 (mevis/lv-vis `videos-2fps/*.mp4`) |

So including any of them crashes training with `FileNotFoundError` on the first such
example. They are dropped until the mp4s land. Note the gaze task we specialize on **is
itself video pointing**, so the 92% gaze group keeps the model training on video every
step — the dropped video rehearsal costs the least in the meantime.

All three dropped groups' shares are redistributed proportionally across the surviving
four (image + text) groups by the renormalization above.

### Renormalized group rates

Surviving paper rates sum to 54.5%. Renormalized shares of the rehearse budget:

| Group                   | renorm share | rate at gaze_w=0.92 |
|-------------------------|-------------:|--------------------:|
| rehearse_captions       |       24.95% |              0.0200 |
| rehearse_image_qa       |       41.65% |              0.0333 |
| rehearse_image_pointing |       16.70% |              0.0134 |
| rehearse_nlp            |       16.70% |              0.0134 |
| **gaze**                |            — |          **0.9200** |

(Verified by `test_mixture.py`; the five rates sum to 1.0.)

## Dataset availability (why these specific datasets)

The datasets in each group are exactly those that not only **construct** (`.load()`
reads their metadata) but actually **traverse** — i.e. their image/video files resolve
and open. We verified this in two passes against the staged, read-only `Molmo2-Data`
tree:

1. a **construction probe** (build each candidate in-container, count examples), and
2. a **traversal probe** (open N random example media files per dataset).

Pass 2 matters: several datasets pass pass 1 but reference media that isn't staged
(the extraction is still running), so they only fail *during training*. Those are
excluded. This list reflects what was traversable at authoring time and is expected to
grow as more mp4s land.

**Token weighting** mirrors the Molmo2 mixture: pointing-message tokens get
`MessageWeight(0.2)` and caption tokens `MessageWeight(0.1)`, so coordinate/caption
supervision is down-weighted the way it was in pretraining-style SFT.

| Group | Datasets (kept — traversable) | Notably excluded (reason) |
|-------|-------------------------------|---------------------------|
| Captions/Long QA | `pixmo_cap`, `pixmo_cap_qa`, `pixmo_cap_qa_as_user_qa`, `pixmo_ask_model_anything` | `molmo2_captions`/`molmo2_syn_captions_qa` (slow build / read-only), `molmo2_human_qa`, `pixmo_multi_image_qa_*` (read-only cache) |
| Image QA | `coco_2014_vqa_multi`, `okvqa`, `chart_qa_weighted`, `ai2_diagram_v2_mix_transparent`, `a_okvqa_mc`, `a_okvqa_da`, `science_qa_img`, CoSyn: `cosyn_chart_exp`, `cosyn_table_exp`, `cosyn_document`, `cosyn_diagram_exp`, `cosyn_math_exp`, multi-image: `mantis_instruct_nlvr2_multi_only` | `text_vqa` (train_images/ not staged), `tally_qa` (images at off-host `/weka/...`), `doc_qa`/`info_qa`/`st_qa` (files absent), `blink` (no train split) |
| Image Pointing | `pixmo_points_high_freq_train`, `pixmo_count_train` | `pixmo_points_train` (None-vs-int bug in this snapshot), `pixmo_multi_points` (0 rows), `cosyn_point` (build timeout) |
| NLP | `tulu4` | — |
| Video QA | *(none usable yet)* | `llava_video_mc/oe`, `clevrer`, `motionbench_train` load metadata but their mp4s are ~0–15% staged → traversal `FileNotFoundError`; most others (`perception_test`, `how2qa`, `star`, `tgif`, `tvqa_*`, `paxion`, …) need manual videos |
| Video Pointing | *(none usable yet)* | `academic_points_clip_63s_2fps` videos (mevis/lv-vis `videos-2fps`) unstaged; `vixmo_points_oversample`/`molmo2_video_point*` need an un-staged `.pkl` |
| Video Tracking | *(none usable)* | entire group: read-only-fs cache write |

### Restore checklist (when video mp4s finish extracting)

1. Re-run the traversal probe (`probe_traverse.py`) against the video datasets; confirm
   example videos resolve (`ok` ≈ `checked`).
2. Add the group back to `rehearse_groups` and its paper rate back to `PAPER_RATES`
   (Video QA 0.182, Video Pointing 0.136, Video Tracking 0.136). The renormalization
   then automatically rebalances all groups.
3. Video Tracking additionally needs a **writable** processed-cache location (its loaders
   write inside `MOLMO_DATA_DIR`), so either pre-build that cache into the staged tree or
   point the cache at writable scratch before re-enabling it.
4. Re-run `test_mixture.py` (rates sum to 1.0) and a short smoke before a full run.

## How to run

Use the launcher; pick the 95/5 gaze split and point at the staged manifest + the
`Molmo2-Data` rehearsal root:

```bash
training/gaze_sft.sh \
  --profile l40 --gpus 4 --cp-degree 4 \
  --name <run> \
  --mixture gaze_specialize --specialize-ratio 0.92 --gaze-objective all \
  --gaze-data-dir <dir with joint/manifest.jsonl + splits/95_05/> \
  --gaze-split-name 95_05 \
  --molmo-data-dir <Molmo2-Data root> \
  --checkpoint <Molmo2-4B-SFT olmo-native ckpt> \
  --seq-len 12288 \
  --wandb-base-url https://far.wandb.io
```

### Sequence length is constrained by the video preprocessor

`--seq-len` **must be ≥ ~11357** for the video model. The preprocessor's worst-case
output (128 frames at the configured pooling) is 11357 tokens, and
`get_output_shapes()` raises
`ValueError: Max sequence length 11357 is greater than preprocessor max token length`
at startup if `seq_len` is smaller — this is a static config check, not a
per-example failure. We use **12288** on L40 (48 GB): comfortably above the floor,
below the 16384 that pushed peak memory to ~45/46 GB under `cp_degree=4`. For longer
sequences prefer raising `--cp-degree` (splits the sequence across ranks) over a
bigger `--seq-len`.

## Verifying the mixture

`third_party/molmo2/test_mixture.py` builds `get_training_mixture("gaze_specialize")`
without loading data and asserts the group weights are correct (gaze = `gaze_w`,
rehearse groups sum to `1 - gaze_w` in the renormalized paper proportions). Run it in
the training image with `GAZE_SPECIALIZE_RATIO` set.

## Smoke validation

This mixture was validated end-to-end on sumedhso-L40S (4×L40 48 GB), gaze split
`95_05` of `full_3hz_min25_max16` (214,758 train clips) against the staged
`Molmo2-Data`:

- All five groups (gaze + 4 rehearse) load and the dataloader **traverses** them
  without `FileNotFoundError` — the gaze group logs `gaze_video_point: 92.00`,
  confirming the 92% weight.
- 8 training steps at `--seq-len 12288 --cp-degree 4 --device-batch-size 1` complete
  and the sharded checkpoint saves to S3 (`runs/l40-smoke-rehearse-1/step8/`).
- **Peak GPU memory ~44.9 GB / 46 GB** (rank 1) — fits 4×L40 with margin; no OOM.

Three bugs were caught and fixed during smoking (each documented above): the
`seq_len < 11357` static check, the read-only-fs tracking loaders, and unstaged video
mp4s surfacing as traversal `FileNotFoundError`s in the video groups + `text_vqa` /
`tally_qa`.

## Editing the mixture

To add a group/dataset back (e.g. once tracking is pre-cached, or more academic
videos are staged): add it to `rehearse_groups` in `get_training_mixture` and, if it
is a new group, give it a rate in `PAPER_RATES` (the renormalization handles the
rest). Re-probe loadability first — a dataset that raises at construction will crash
the dataloader build.
