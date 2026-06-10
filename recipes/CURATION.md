# Curation Pipeline (recipes → extract → overlay → smoke manifest)

End-to-end workflow for turning the read-only raw datasets on the NFS host into
viewer-ready, gaze+annotation-overlaid sample episodes. All steps are exposed
through the single `gaze` CLI under the `curate` command group, alongside the
existing `datasets`, `rectify`, `validate`, `split`, `serve`, and `s3` groups,
and the planned manifest-producing task.

## Architecture

- **Source data is read-only** on the remote NFS host (`sumedhso-L40S`,
  `/nfs/colbhben/gaze/unprocessed/<slug>/`). Pulled per-episode via ssh/scp.
- **The local machine is the processing host** — it has `ffprobe`/`ffmpeg`
  (video metadata + overlay encoding), `numpy`/`pandas` (`.npy` gaze / `.pkl`
  narrations), and `projectaria_tools` (Aria CPF gaze → pixel projection). The
  remote has none of these, so all parsing/projection/rendering runs locally.
- **Uploads go through the remote** (`aws s3` on the remote host): local AWS
  access is read-denied, so the smoke manifest is `scp`'d to the remote and
  `aws s3 sync`'d to the bucket from there.

## The three layers (see README.md)

1. **Observed** — what's on disk (the existing inspect/watch manifest).
2. **Recipe** — `recipes/<slug>.json`: the soft parse/extract decisions
   (episode enumeration, file selection, native gaze format + projection,
   multi-channel annotations, `epoch_sync`). Validated by
   `scripts/validate_recipes.py`.
3. **Resolved** — the per-episode bundle the extractor emits, and the smoke
   manifest the viewer serves.

## CLI

```sh
# 1. Extract one episode's video metadata + gaze + annotation channels.
#    Omit --episode to use recipes/_sample_episodes.json.
gaze curate extract --dataset nymeria
gaze curate extract --dataset holoassist --episode R0027-12-GoPro

# 2. Render a gaze + annotation overlay clip (projects gaze per frame, draws
#    the gaze dot + active annotation captions, reconciling clocks via epoch_sync).
gaze curate overlay --dataset nymeria --max-seconds 20

# 3. Assemble a viewer-ready smoke manifest from extracted bundles + overlays.
gaze curate build-smoke --out-root /tmp/gaze_smoke_manifest

# 4. Upload it to S3 (via the remote host).
gaze curate upload-smoke --root /tmp/gaze_smoke_manifest          # --dry-run to preview

# Or end-to-end build (+ optional upload) from already-extracted artifacts:
gaze curate smoke --upload

# 5. View it locally (the overlay mp4 is each episode's "video"):
gaze serve --canonical-root /tmp/gaze_smoke_manifest
```

For datasets with large source mp4s (nymeria ~600 MB–1.2 GB), `overlay` pulls
the file, trims to `--max-seconds`, renders, and deletes the source. Keep
`--max-seconds` small for smoke samples.

## Smoke manifest layout (viewer-compatible)

```
<root>/
  manifest.jsonl              # episode index
  manifest.parquet[.jsonl]    # table form the server reads (id, dataset, episode_id, ...)
  smoke_report.json
  episodes/<dataset>/<episode_id>/
    episode.json              # {dataset, episode_id, files{video,gaze,annotations}, video_meta, gaze_meta, annotation_channels}
    overlay.mp4               # gaze + annotation overlay (served as "video")
    side_by_side.mp4          # GT|ours, where ground truth is available (nymeria)
    gaze.jsonl                # reconciled gaze rows (video-zero seconds)
    annotations.jsonl         # flattened raw segments across channels
    bundle.json               # full extraction bundle (provenance)
```

Published at `s3://far-research-internal/colbhben/gaze/unprocessed/smoke_manifest/`.

## Gaze projection per dataset

| dataset | gaze space | how the overlay gets 2D pixels |
|---|---|---|
| ego-exo4d | pixel_2d (1408²) | already projected; drawn directly |
| egtea | pixel_2d (1280×960) | drawn directly (gaze sliced from session by clip frame range) |
| egome | pixel_2d (1280×960) | drawn directly |
| egoexolearn | normalized_2d | x·W, y·H |
| nymeria, hd-epic | cpf_angular | `projectaria_tools.get_gaze_vector_reprojection` (camera-rgb calib from `online_calibration.jsonl`, `make_upright=True`, scale ×mp4_w/2880, per-sample depth_m) — **validated** |
| holoassist | head_ray_3d | psi pinhole: 3D ray → camera via `Pose_sync.txt` 4×4 → `Video/Intrinsics.txt` (896×504) |

## Clock reconciliation (`epoch_sync`)

Each recipe declares how video frames, gaze samples, and annotation channels
map onto one **video-zero-seconds** clock (verified against video duration):

- **ego-exo4d / nymeria / hd-epic** gaze is absolute device-clock → subtract the
  first gaze sample (which == mp4 frame 0). nymeria annotations share that device
  clock; hd-epic annotations are already video-zero (and ship a per-frame
  `*_mp4_to_vrs_time_ns.csv` for exact timing).
- **egtea** annotations are session-absolute ms → subtract the clip start.
- **egoexolearn / egome / holoassist** are already video-zero (no transform).

## Ground-truth comparison

For the Aria datasets, our projection uses the same `get_gaze_vector_reprojection`
function Meta's projectaria-explorer renders with. To eyeball-compare against the
official overlay at `explorer.projectaria.com/nymeria/<take>`, set the `st=` query
param to device-clock seconds `= (first_gaze_ts_us + frame·1e6/fps)/1e6`. The
nymeria `side_by_side.mp4` pairs our overlay with a GT-reference panel carrying
those instructions. Self-consistency on nymeria: 100% in-frame over sampled
frames, 0 teleports, personalized-vs-general gaze agree to <2% of frame width.

## Validation

```sh
python scripts/validate_recipes.py                       # recipes vs schema + cross-field
.venv/bin/python -m pytest tests/test_curate.py tests/test_overlay.py -q
.venv/bin/python -m unittest discover -s tests           # full suite
```
