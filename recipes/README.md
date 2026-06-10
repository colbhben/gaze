# Dataset Curation Recipes

Each `recipes/<slug>.json` captures the **soft human decisions** about how to parse and
extract a dataset that the observed manifest cannot infer on its own. The manifest
generator reads the matching recipe and folds a resolved `processing` block into each
dataset's manifest entry, so the downstream rectifier consumes one deterministic artifact.

Three layers, one artifact:

1. **Observed** — machine-generated (what is on disk; inferred structure). Existing manifest.
2. **Recipe** — this directory; hand-authored per dataset; the soft choices below.
3. **Resolved** — the validated merge the rectifier reads. Recipe overrides heuristics;
   where the recipe is silent, observed inference still fills in.

`_defaults.json` holds settings shared by all datasets (targets, emit policy, metadata
fields, tool requirements). A recipe may override any default.

Validate with: `python scripts/validate_recipes.py` (stdlib only; checks every recipe
against `recipe.schema.json` and applies cross-field rules; cross-checked against the
`jsonschema` library as an oracle).

## Dataset status (gaze space, projection, and what's validated)

All facts below were confirmed by reading real bytes on the `/nfs` mount; projection
specifics were validated on the local toolchain (ffprobe/ffmpeg/numpy/pandas/projectaria_tools).

| dataset | episode | gaze space | projection method | metadata notes |
|---|---|---|---|---|
| ego-exo4d | take | pixel_2d (1408²) | already_2d | gaze `_2d.csv` pre-projected; prefer personalized→general |
| egoexolearn | ego video_uid | normalized_2d | normalize_by_dims | gaze `(N,3)` x,y,validity @30fps; mask invalid (0.5,0.5) |
| egome | `_ego` stem | pixel_2d **1280×960** (resolved) | normalize_by_dims | fps **varies per take** (28–30, ffprobe per-episode); 70-col Tobii+IMU |
| egtea | clip | pixel_2d (1280×960) | normalize_by_dims | 24 fps; two BeGaze variants; gaze sliced from session by frame range |
| holoassist | take | head_ray_3d | psi_pinhole_ray (**unblocked**) | intrinsics `cam_info/.../Video/Intrinsics.txt` = 896×504 (no rescale); fps per-take (24.46/30); gaze grid 30 Hz decoupled from mp4 fps → map by time |
| nymeria | take | cpf_angular | projectaria_cpf (**validated**) | `make_upright=true`, scale ×(mp4_w/2880), per-sample depth_m; calib from `online_calibration.jsonl` alone |
| hd-epic | take | cpf_angular | projectaria_cpf | same Aria path as nymeria; calib via `vrs_to_multi_slam.json` deref; exact frame timing from `*_mp4_to_vrs_time_ns.csv` |

`projection.validated: true` marks a method checked end-to-end (nymeria: 5 frames landed on
correct targets). holoassist/hd-epic projection is fully specified and feasible but the
end-to-end pixel check vs the projectaria-explorer ground-truth overlay is the remaining
validation step.

---

## Episode model

Every episode anchors on the **egocentric RGB video**. Per `_defaults.emit_policy`, an
episode is emitted only when it has video **and** gaze **and** ≥1 annotation channel;
a missing piece warns and is skipped, never errors.

## Field vocabulary

Top level of a recipe:

| Key | Meaning |
|---|---|
| `dataset` | slug (must match filename and a `DATASETS.md` slug) |
| `root` | dataset root under the unprocessed layout, relative to `unprocessed/` |
| `episode` | how to enumerate episodes and derive their ids (see **episode**) |
| `video` | how to locate the ego RGB mp4 (see **modality.select**) |
| `gaze` | how to locate + interpret gaze (select + `gaze_format`) |
| `annotations` | ordered list of named annotation channels (see **annotation channel**) |
| `metadata` | per-dataset metadata hints (shipped sources, sampling) |
| `overrides` | per-dataset config overrides (e.g. `target_hz`) |
| `provenance` | verbatim notes, resolved decisions, open questions |

### episode

```jsonc
"episode": {
  "enumerate": "video_glob | take_dir | annotation_keys | id_list",
  "glob": "<glob relative to root, when enumerate=video_glob/take_dir>",
  "id_from": { "rule": "path_component|filename_regex|filename_stem|literal", "level": 0, "regex": "...", "group": "take" },
  "id_sanitize": true,
  "filter": { "include_regex": "...", "exclude_regex": "...", "comment": "..." },
  "bridge": { "file": "<rel path>", "maps": "id_a<->id_b", "comment": "join key bridge, e.g. take_name<->take_uid" }
}
```

- `enumerate` picks the iteration source: a video glob, per-take directories, the keys of
  an annotation file, or an explicit id list.
- `id_from` derives the canonical episode id from each enumerated item.
- `filter` drops items by regex (e.g. egome `_ego` only; egoexolearn ego-`view` is handled
  on the annotation join, not here).
- `bridge` records an id↔id mapping file when channels key differently (ego-exo4d).

### modality.select  (used by `video`, `gaze`, and each annotation channel)

How to resolve one file per episode when several may match:

```jsonc
"select": {
  "path_template": "<rel path with {episode_id} / {take} / {prefix} placeholders>",
  "glob": "<glob within the episode scope>",
  "prefer": ["personalized_eye_gaze_2d.csv", "general_eye_gaze_2d.csv"],
  "pick": "first_existing | largest | smallest | single | invariant_suffix",
  "invariant_suffix": "214-1.mp4",
  "from_metadata": { "file": "...", "field": "frame_aligned_videos.*.rgb.relative_path" },
  "missing": "warn|skip|error"
}
```

- `prefer` is an ordered fallback list (first existing wins).
- `pick=invariant_suffix` selects by a stable suffix when a prefix varies (ego-exo4d
  `<aria>_214-1.mp4`).
- `from_metadata` resolves the path from a metadata file field rather than disk globbing.

### gaze_format

Declarative description of the native gaze representation **plus** a projection recipe.
Per the project decision, gaze is preserved **native**; the manifest records how to
project to normalized-2D, and feasibility, without forcing projection at manifest time.

```jsonc
"gaze_format": {
  "reader": "csv|npy|whitespace_txt|begaze_txt",
  "coordinate_space": "pixel_2d | normalized_2d | cpf_angular | head_ray_3d",
  "frame_dims": [1408, 1408],            // when pixel_2d, for normalization
  "columns": { "x": "x", "y": "y" },     // source->canonical channel map (space-specific)
  "time": { "source": "tracking_timestamp_us", "units": "us|ms|s|frame_index", "fps": 30, "epoch": "absolute|video_zero" },
  "validity": { "column": "validity", "valid_when": "==1", "invalid_placeholder": [0.5, 0.5] },
  "extra_channels": ["depth_m", "left_yaw_rads_cpf", "..."],   // preserved as-is
  "projection": {
    "feasible": true,
    "method": "already_2d | normalize_by_dims | projectaria_cpf | psi_pinhole_ray",
    "needs": ["online_calibration.jsonl"],
    "calibration": { "file": "<rel path or template>", "format": "aria_online_calib|psi_intrinsics_txt" },
    "extrinsics": { "file": "<rel path>", "format": "psi_pose_sync_4x4" },
    "notes": "..."
  },
  "variants": [ { "reader": "begaze_txt", "version": "3.1", "match": "## Version: BeGaze 3.1", "columns": {...} } ]
}
```

`variants` lets one gaze source carry multiple on-disk layouts selected at parse time
(egtea's two BeGaze exports).

### annotation channel

`annotations` is an **ordered list**; each entry is one named channel preserved
separately (per the multi-channel requirement). Segments are kept **raw** (lossless);
the manifest also records per-channel timing stats.

```jsonc
{
  "name": "atomic",                       // channel name, unique within dataset
  "select": { ... },                      // how to find the annotation file(s)
  "reader": "csv|json|pandas_pickle|json_by_key",
  "kind": "interval | point",             // interval = start+end; point = single timestamp
  "time": { "start": "start_sec", "end": "end_sec", "point": "timestamp", "units": "s|ms|us|filetime_100ns", "epoch": "video_zero|absolute" },
  "text": { "primary": "narration_en", "compose": ["verb", "noun"], "strip_prefix_regex": "^[A-Z]:" },
  "extras": ["main_actions", "Action Correctness"],   // structured fields preserved alongside text
  "join": { "by": "episode_id | key | stem", "key_field": "video_uid", "key_in_file": "take_uid", "select_path": "data.annotations.{key}" },
  "filter": { "drop_where": "label=='Conversation'", "keep_where": "view=='ego'", "denylist_file": "..." },
  "split_field": "subset",                // optional: where train/val/test lives
  "missing": "warn"
}
```

- `kind=point` covers single-timestamp events (ego-exo4d atomic); `interval` covers spans.
- `text.compose` builds text from multiple columns (e.g. holoassist fine Verb+Adjective+Noun);
  `text.strip_prefix_regex` removes shipped prefixes (egome `C:`/`T:`/`F:`) while the raw
  value is retained.
- `extras` preserves structured sidecar fields (holoassist correctness, hd-epic main_actions).
- `join` describes how a row links to an episode; `filter.drop_where`/`keep_where` are simple
  declarative predicates; `denylist_file` drops ids (hd-epic erratum).

### provenance

```jsonc
"provenance": {
  "notes": "<your verbatim notes>",
  "decisions": ["resolved choice + why"],
  "corrections": ["where on-disk reality differed from the notes"],
  "open_questions": ["anything still needing a human/runtime to confirm"]
}
```

This is where the rigor extends to *interpretation*: the original intent, the resolved
decision, the deltas found on disk, and what remains unverified.

## Declarative-only

Per project decision, recipes are **pure data** — no embedded code or expressions beyond
the small enumerated predicates above (`drop_where`, `valid_when`, regexes). When a dataset
needs a transform the vocabulary can't express, we extend the schema with a new declarative
construct (and bump `recipe_version`) rather than embedding logic in the recipe.
