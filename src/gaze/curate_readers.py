"""Per-dataset gaze / annotation readers + projection + episode assembly.

This module sits on top of the stable contract in ``curate.py`` (recipe loading,
``Puller``, dataclasses, ``ffprobe_video``, ``to_seconds``, ``annotation_stats``,
``compose_text``, ``_dotted_get``).  It implements:

  * GAZE readers, keyed by ``gaze_format.reader``:
      csv | npy | whitespace_txt | begaze_txt
  * ANNOTATION readers, keyed by ``reader``:
      csv | json_by_key | pandas_pickle
  * GAZE -> 2D projection, keyed by ``gaze_format.projection.method``:
      already_2d | normalize_by_dims | projectaria_cpf | psi_pinhole_ray
  * ``extract_episode(slug, episode_id, puller, **kw) -> EpisodeBundle``:
      resolves the video/gaze/annotation files for one episode per the recipe
      ``select`` blocks, pulls them, ffprobes the video, parses gaze + each
      annotation channel, computes stats, and enforces the emit policy.

Heavy deps (numpy / pandas / projectaria_tools) are imported lazily inside the
readers that need them so the import path stays light.
"""
from __future__ import annotations

import fnmatch
import json
import math
import re
from pathlib import Path
from typing import Any, Callable

from . import curate
from .curate import (
    AnnotationChannel,
    EpisodeBundle,
    GazeTable,
    Puller,
    VideoMeta,
    annotation_stats,
    compose_text,
    ffprobe_video,
    load_recipe,
    to_seconds,
    _dotted_get,
)


# --------------------------------------------------------------------------- #
# Dataset-level filters (recipe `dataset_filters` block).
# Applied during episode selection (enumeration) and training-manifest builds.
# --------------------------------------------------------------------------- #
def dataset_filters(slug: str, recipe: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the recipe's `dataset_filters` block (or {} if absent)."""
    rec = recipe if recipe is not None else load_recipe(slug)
    return rec.get("dataset_filters") or {}


def is_culled(slug: str, recipe: dict[str, Any] | None = None) -> bool:
    """True if the dataset is excluded from all generation (recipe retained)."""
    return bool(dataset_filters(slug, recipe).get("cull"))


def episode_excluded(slug: str, episode_id: str, recipe: dict[str, Any] | None = None) -> bool:
    """True if the episode id matches any `exclude_episode_globs` pattern."""
    globs = dataset_filters(slug, recipe).get("exclude_episode_globs") or []
    return any(fnmatch.fnmatch(episode_id, pat) for pat in globs)


def filter_episode_ids(slug: str, episode_ids: list[str], recipe: dict[str, Any] | None = None) -> list[str]:
    """Drop culled-dataset (all) and glob-excluded episode ids. Reusable by the
    smoke driver and the full-dataset enumerator."""
    rec = recipe if recipe is not None else load_recipe(slug)
    if is_culled(slug, rec):
        return []
    return [e for e in episode_ids if not episode_excluded(slug, e, rec)]


def max_gaze_gap_exceeded(gaze_times_video: list[float | None], valid_flags: list[bool], max_gap_s: float) -> tuple[bool, float]:
    """True if the gap between any two consecutive VALID gaze samples exceeds
    `max_gap_s`. Returns (exceeded, observed_max_gap_s). Times are the reconciled
    video-clock gaze times; pair each with its validity flag."""
    valids = sorted(
        t for t, ok in zip(gaze_times_video, valid_flags) if t is not None and ok
    )
    max_gap = 0.0
    for a, b in zip(valids, valids[1:]):
        gap = b - a
        if gap > max_gap:
            max_gap = gap
    return (max_gap > max_gap_s, max_gap)


# --------------------------------------------------------------------------- #
# Small declarative-predicate evaluator (drop_where / keep_where / valid_when)
# --------------------------------------------------------------------------- #
_PRED_RE = re.compile(r"^\s*([A-Za-z0-9_.\- ]+?)\s*(==|!=|>=|<=|>|<)\s*(.+?)\s*$")


def _coerce_literal(token: str) -> Any:
    token = token.strip()
    if (token.startswith("'") and token.endswith("'")) or (
        token.startswith('"') and token.endswith('"')
    ):
        return token[1:-1]
    low = token.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in ("none", "null"):
        return None
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        pass
    return token


def eval_predicate(row: dict[str, Any], expr: str | None) -> bool:
    """Evaluate a simple ``field <op> literal`` predicate against a row dict.

    Supports ==, !=, >=, <=, >, <. The field may be dotted (``attributes.X``).
    Missing fields compare as None. Returns True when ``expr`` is falsy.
    """
    if not expr:
        return True
    m = _PRED_RE.match(expr)
    if not m:
        raise ValueError(f"unparseable predicate: {expr!r}")
    field, op, rhs_tok = m.group(1).strip(), m.group(2), m.group(3)
    lhs = _dotted_get(row, field)
    rhs = _coerce_literal(rhs_tok)
    if op == "==":
        return _eq(lhs, rhs)
    if op == "!=":
        return not _eq(lhs, rhs)
    # numeric comparisons
    try:
        lf, rf = float(lhs), float(rhs)
    except (TypeError, ValueError):
        return False
    if op == ">":
        return lf > rf
    if op == "<":
        return lf < rf
    if op == ">=":
        return lf >= rf
    if op == "<=":
        return lf <= rf
    raise ValueError(f"unknown operator {op!r}")


def _eq(a: Any, b: Any) -> bool:
    if isinstance(b, bool):
        return bool(a) == b
    if a is None:
        return b is None
    if isinstance(b, (int, float)) and not isinstance(b, bool):
        try:
            return float(a) == float(b)
        except (TypeError, ValueError):
            return False
    return str(a) == str(b)


def _eval_valid(value: Any, valid_when: str) -> bool:
    """valid_when is like '==1', '>=0' (no left-hand field)."""
    m = re.match(r"^\s*(==|!=|>=|<=|>|<)\s*(.+?)\s*$", valid_when)
    if not m:
        raise ValueError(f"unparseable valid_when: {valid_when!r}")
    return eval_predicate({"_v": value}, f"_v {m.group(1)} {m.group(2)}")


# --------------------------------------------------------------------------- #
# Episode-id token decomposition (session / participant / take_name ...)
# --------------------------------------------------------------------------- #
def _episode_tokens(slug: str, episode_id: str, extra: dict[str, Any]) -> dict[str, str]:
    """Build the substitution dict used to fill recipe ``path_template`` strings."""
    # Common aliases: several recipes name the episode-key placeholder differently
    # (video_uid, take_name, stem). Default them all to the episode id.
    tok: dict[str, str] = {
        "episode_id": episode_id,
        "video_uid": episode_id,
        "stem": episode_id,
    }
    # The sample file may carry session / participant / take_name overrides.
    for k in ("session", "participant", "take_name", "take_uid", "take_name", "slam_index"):
        if k in extra and extra[k] is not None:
            tok[k] = str(extra[k])
    # Per-dataset derivations when not supplied.
    if slug == "egtea" and "session" not in tok:
        # clip id = <SESSION>-<startMs>-<endMs>-F<startFrame>-F<endFrame>
        tok["session"] = re.sub(r"-\d+-\d+-F\d+-F\d+$", "", episode_id)
    if slug == "hd-epic" and "participant" not in tok:
        m = re.match(r"^(P\d+)", episode_id)
        if m:
            tok["participant"] = m.group(1)
    if slug == "ego-exo4d" and "take_name" not in tok:
        tok["take_name"] = episode_id
    return tok


def _fill(template: str, tok: dict[str, str]) -> str:
    out = template
    for k, v in tok.items():
        out = out.replace("{" + k + "}", str(v))
    return out


# --------------------------------------------------------------------------- #
# File resolution (the recipe ``select`` block)
# --------------------------------------------------------------------------- #
def resolve_file(
    puller: Puller,
    root: str,
    select: dict[str, Any],
    tok: dict[str, str],
    *,
    metadata_resolver: Callable[[dict[str, Any]], str | None] | None = None,
) -> str | None:
    """Return the source-root-relative path of the chosen file, or None.

    Honors prefer (first_existing), glob, path_template/single, invariant_suffix,
    and from_metadata. ``root`` is the dataset root relative to ``unprocessed/``.
    """
    def rel(p: str) -> str:
        return f"{root}/{p}".replace("//", "/")

    pick = select.get("pick")

    # 1. from_metadata (ego-exo4d) -- handled by caller via metadata_resolver
    if select.get("from_metadata") and metadata_resolver is not None:
        got = metadata_resolver(select["from_metadata"])
        if got:
            return rel(_fill(got, tok)) if not got.startswith(root) else got

    # 2. prefer list (first existing wins)
    if select.get("prefer"):
        base = _fill(select.get("path_template", ""), tok).rstrip("/")
        for cand in select["prefer"]:
            candrel = rel(f"{base}/{cand}") if base else rel(cand)
            if puller.exists(candrel):
                return candrel
        return None

    # 3. glob within the episode scope
    if select.get("glob"):
        pattern = rel(_fill(select["glob"], tok))
        matches = puller.glob(pattern)
        if not matches:
            return None
        if pick == "invariant_suffix":
            suff = select.get("invariant_suffix", "")
            for m in matches:
                if m.endswith(suff):
                    return m
            return matches[0]
        if pick in ("single", None, "first_existing"):
            return sorted(matches)[0]
        if pick == "largest":
            return matches[-1]
        return matches[0]

    # 4. path_template direct (single)
    if select.get("path_template"):
        cand = rel(_fill(select["path_template"], tok))
        if puller.exists(cand):
            return cand
        return None

    return None


# =========================================================================== #
# GAZE READERS
# =========================================================================== #
def read_gaze_csv(local: Path, gf: dict[str, Any], fps: float | None) -> GazeTable:
    """Header CSV -> GazeTable. Maps columns per gaze_format.columns, time via to_seconds."""
    import csv as _csv

    cols = gf.get("columns", {})
    time_spec = gf.get("time", {})
    validity = gf.get("validity") or {}
    extra = gf.get("extra_channels") or []
    space = gf["coordinate_space"]

    rows: list[dict[str, Any]] = []
    valid_count = 0
    with open(local, newline="", encoding="utf-8", errors="replace") as fh:
        reader = _csv.DictReader(fh)
        for r in reader:
            out: dict[str, Any] = {}
            for canon, src in cols.items():
                out[canon] = _num(r.get(str(src)))
            t = r.get(str(time_spec.get("source")))
            out["t_s"] = to_seconds(t, time_spec.get("units", "s"), fps)
            is_valid = True
            if validity.get("column") is not None and validity.get("valid_when"):
                raw = r.get(str(validity["column"]))
                is_valid = _eval_valid(_num(raw), validity["valid_when"])
            out["valid"] = bool(is_valid)
            for ec in extra:
                out[str(ec)] = r.get(str(ec))
            if is_valid:
                valid_count += 1
            rows.append(out)
    return _finalize_gaze(rows, space, list(cols.keys()), gf, valid_count)


def read_gaze_npy(local: Path, gf: dict[str, Any], fps: float | None) -> GazeTable:
    """numpy (N,3) load: col0=x_norm, col1=y_norm, col2=validity. time = frame/fps."""
    import numpy as np

    arr = np.load(local)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    cols = gf.get("columns", {})
    xcol = int(cols.get("x", 0))
    ycol = int(cols.get("y", 1))
    validity = gf.get("validity") or {}
    vcol = int(validity["column"]) if validity.get("column") is not None else None
    valid_when = validity.get("valid_when")
    time_spec = gf.get("time", {})
    use_fps = time_spec.get("fps") or fps
    space = gf["coordinate_space"]

    rows: list[dict[str, Any]] = []
    valid_count = 0
    for i in range(arr.shape[0]):
        x = float(arr[i, xcol]) if xcol < arr.shape[1] else None
        y = float(arr[i, ycol]) if ycol < arr.shape[1] else None
        is_valid = True
        if vcol is not None and vcol < arr.shape[1] and valid_when:
            is_valid = _eval_valid(float(arr[i, vcol]), valid_when)
        out = {
            "x": x,
            "y": y,
            "t_s": (i / use_fps) if use_fps else None,
            "frame": i,
            "valid": bool(is_valid),
        }
        if vcol is not None and vcol < arr.shape[1]:
            out["validity"] = float(arr[i, vcol])
        if is_valid:
            valid_count += 1
        rows.append(out)
    return _finalize_gaze(rows, space, ["x", "y"], gf, valid_count)


def read_gaze_whitespace_txt(local: Path, gf: dict[str, Any], fps: float | None) -> GazeTable:
    """No-header whitespace/tab split, integer column indices (holoassist Eyes_sync.txt)."""
    cols = gf.get("columns", {})
    time_spec = gf.get("time", {})
    validity = gf.get("validity") or {}
    extra = gf.get("extra_channels") or []
    space = gf["coordinate_space"]

    rows: list[dict[str, Any]] = []
    valid_count = 0
    with open(local, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\r\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = re.split(r"[\t ]+", line.strip())
            out: dict[str, Any] = {}
            for canon, idx in cols.items():
                i = int(idx)
                out[canon] = _num(parts[i]) if i < len(parts) else None
            ts = time_spec.get("source")
            if ts is not None:
                i = int(ts)
                out["t_s"] = to_seconds(parts[i] if i < len(parts) else None,
                                        time_spec.get("units", "s"), fps)
            is_valid = True
            if validity.get("column") is not None and validity.get("valid_when"):
                i = int(validity["column"])
                raw = parts[i] if i < len(parts) else None
                is_valid = _eval_valid(_num(raw), validity["valid_when"])
            out["valid"] = bool(is_valid)
            for ec in extra:
                i = int(ec)
                out[f"col{i}"] = _num(parts[i]) if i < len(parts) else None
            if is_valid:
                valid_count += 1
            rows.append(out)
    return _finalize_gaze(rows, space, list(cols.keys()), gf, valid_count)


def read_gaze_begaze_txt(
    local: Path, gf: dict[str, Any], fps: float | None, frame_range: tuple[int, int] | None
) -> GazeTable:
    """egtea BeGaze export. Branch on the '## Version:' header (3.1 vs 3.4) per variants.

    Slices to the clip's [startFrame, endFrame] range (frame_range) when given.
    """
    text = local.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    # Determine variant from the version header line.
    version_line = next((ln for ln in lines if "Version:" in ln), "")
    norm = re.sub(r"\s+", " ", version_line).strip()
    chosen = None
    for variant in gf.get("variants", []):
        match = re.sub(r"\s+", " ", variant["match"]).strip()
        if match in norm:
            chosen = variant
            break
    if chosen is None:
        chosen = {"columns": gf.get("columns", {}), "time": gf.get("time", {})}

    cols = chosen.get("columns", gf.get("columns", {}))
    time_spec = chosen.get("time", gf.get("time", {}))
    use_fps = time_spec.get("fps") or fps
    space = gf["coordinate_space"]

    # Data lines: drop '##' comment block AND the in-file column header row
    # (which starts with a non-numeric token like 'Time').
    data_lines = []
    for ln in lines:
        if ln.startswith("##") or not ln.strip():
            continue
        first = re.split(r"\t", ln.strip())[0]
        if not re.match(r"^-?\d", first):  # header row (e.g. 'Time')
            continue
        data_lines.append(ln)

    units = time_spec.get("units", "frame_index")

    # For the microsecond-`Time`-column variant (BeGaze 3.4, e.g. 3.4.46: 27 tab cols,
    # col0 `Time` in us @ ~30Hz), the session frame index is derived from the elapsed
    # time off the FIRST sample: frame = round((Time - Time0)/1e6 * fps). The clip
    # filename frame range (F<start>-F<end>) is at this `fps` (24). t0 is read once.
    t0_us = None
    if units == "us_to_frame":
        for ln in data_lines:
            p0 = ln.split("\t", 1)[0]
            v = _num(p0)
            if v is not None:
                t0_us = v
                break

    rows: list[dict[str, Any]] = []
    valid_count = 0
    for ln in data_lines:
        parts = ln.rstrip("\r\n").split("\t")
        out: dict[str, Any] = {}
        for canon, idx in cols.items():
            i = int(idx)
            out[canon] = parts[i] if -len(parts) <= i < len(parts) else None
            if canon in ("x", "y"):
                out[canon] = _num(out[canon])
        fi = time_spec.get("source")
        if units == "us_to_frame":
            # source col = the us `Time`; t_s = (Time - Time0)/1e6; frame @ fps.
            i = int(fi) if fi is not None else 0
            raw = _num(parts[i]) if (i is not None and -len(parts) <= i < len(parts)) else None
            if raw is not None and t0_us is not None:
                ts = (raw - t0_us) / 1e6
                out["t_s"] = round(ts, 6)
                out["frame"] = int(round(ts * (use_fps or 24)))
            else:
                out["t_s"] = None
                out["frame"] = None
        else:
            if fi is not None:
                i = int(fi)
                raw_t = parts[i] if -len(parts) <= i < len(parts) else None
                out["t_s"] = to_seconds(raw_t, units, use_fps)
            # frame number for slicing (3.1: integer frame col)
            if units == "frame_index":
                fr = _num(out.get("frame"))
                out["frame"] = int(fr) if fr is not None else None
            else:
                out["frame"] = int(round((out.get("t_s") or 0.0) * (use_fps or 24)))
        out["valid"] = True
        rows.append(out)

    # Slice to clip frame range.
    notes = []
    if frame_range is not None:
        lo, hi = frame_range
        sliced = [r for r in rows if r.get("frame") is not None and lo <= r["frame"] <= hi]
        notes.append(f"sliced session gaze to clip frames [{lo},{hi}]: {len(sliced)}/{len(rows)} rows")
        # rebase time to the clip start so t_s starts ~0
        if sliced:
            t0 = min(r["t_s"] for r in sliced if r.get("t_s") is not None)
            for r in sliced:
                if r.get("t_s") is not None:
                    r["t_s"] = round(r["t_s"] - t0, 6)
        rows = sliced
    valid_count = len(rows)
    gt = _finalize_gaze(rows, space, list(cols.keys()), gf, valid_count)
    gt.notes.extend(notes)
    return gt


GAZE_READERS: dict[str, Callable] = {
    "csv": read_gaze_csv,
    "npy": read_gaze_npy,
    "whitespace_txt": read_gaze_whitespace_txt,
    "begaze_txt": read_gaze_begaze_txt,
}


def _finalize_gaze(
    rows: list[dict[str, Any]], space: str, channels: list[str], gf: dict[str, Any], valid_count: int
) -> GazeTable:
    times = [r["t_s"] for r in rows if r.get("t_s") is not None]
    dur = (max(times) - min(times)) if len(times) >= 2 else None
    hz = (len(times) / dur) if dur and dur > 0 else None
    n = len(rows)
    valid_fraction = (valid_count / n) if n else None
    return GazeTable(
        coordinate_space=space,
        columns=channels,
        rows=rows,
        hz=round(hz, 4) if hz else None,
        sample_count=n,
        duration_s=round(dur, 6) if dur else None,
        valid_fraction=round(valid_fraction, 6) if valid_fraction is not None else None,
        projection=gf.get("projection"),
    )


def _num(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


# =========================================================================== #
# ANNOTATION READERS
# =========================================================================== #
def _segment(
    *,
    start: float | None = None,
    end: float | None = None,
    point: float | None = None,
    text: str | None = None,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "start_s": start,
        "end_s": end,
        "point_s": point,
        "text": text,
        "extras": extras or {},
    }


def _gather_extras(row: dict[str, Any], extras_spec: list[str]) -> dict[str, Any]:
    out = {}
    for key in extras_spec or []:
        out[key] = _dotted_get(row, key)
    return out


def _path_get(obj: Any, key: str) -> Any:
    """Like curate._dotted_get but also supports integer list indices in the path
    (e.g. 'Step timestamp.0' -> obj['Step timestamp'][0]). Falls back to _dotted_get."""
    if isinstance(obj, dict) and key in obj:
        return obj[key]
    cur: Any = obj
    for part in key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, (list, tuple)) and part.lstrip("-").isdigit():
            i = int(part)
            cur = cur[i] if -len(cur) <= i < len(cur) else None
        else:
            return _dotted_get(obj, key)
    return cur


def _col_get(row: dict[str, Any], name: str) -> Any:
    """Tolerant column lookup: exact match, then a header whose text before the first
    '(' equals the requested name (e.g. 'Clip Prefix' -> 'Clip Prefix (Unique)')."""
    if name in row:
        return row[name]
    target = name.strip().lower()
    for k, v in row.items():
        if k is None:
            continue
        base = k.split("(")[0].strip().lower()
        if base == target:
            return v
    return None


def _normalize_csv_row(row: dict[str, Any]) -> dict[str, Any]:
    """Add base-name aliases (text before '(') so recipe column names resolve even when
    the on-disk header carries a parenthetical suffix."""
    out = dict(row)
    for k, v in row.items():
        if k is None:
            continue
        base = k.split("(")[0].strip()
        if base and base != k and base not in out:
            out[base] = v
    return out


def read_anno_csv(
    local: Path, channel: dict[str, Any], episode_id: str, tok: dict[str, str],
    join_value: str | None = None,
) -> list[dict[str, Any]]:
    """CSV annotation reader (egoexolearn fine, egtea action, nymeria activity/atomic).

    Supports semicolon delimiter (egtea), join by row_field or stem, keep/drop filters.
    ``join_value`` overrides the value matched against ``join.key_field`` for row_field
    joins (e.g. egtea's clip-prefix = episode_id minus the trailing -F<start>-F<end>).
    """
    if join_value is None:
        join_value = episode_id
    import csv as _csv

    # Sniff delimiter: peek the first line.
    head = local.read_text(encoding="utf-8", errors="replace").splitlines()
    # egtea action_labels.csv has a leading '# ...' comment header line then ';'-delim
    delim = ","
    data_start = 0
    if head and head[0].lstrip().startswith("#") and ";" in head[0]:
        delim = ";"
        data_start = 1  # the '# ...' line IS the header (strip the leading '# ')
    elif head and head[0].count(";") > head[0].count(","):
        delim = ";"

    join = channel.get("join", {})
    by = join.get("by", "row_field")
    key_field = join.get("key_field")
    time_spec = channel.get("time", {})
    text_spec = channel.get("text", {})
    extras_spec = channel.get("extras", [])
    flt = channel.get("filter", {})

    with open(local, newline="", encoding="utf-8", errors="replace") as fh:
        if data_start == 1:
            header_line = head[0].lstrip("#").strip()
            fieldnames = [c.strip() for c in header_line.split(delim)]
            reader = _csv.DictReader(
                fh, fieldnames=fieldnames, delimiter=delim
            )
            next(reader)  # skip the comment/header line itself
        else:
            reader = _csv.DictReader(fh, delimiter=delim)
            reader.fieldnames = [c.strip() for c in (reader.fieldnames or [])]
        segments = []
        for raw in reader:
            r = {(k.strip() if k else k): (v.strip() if isinstance(v, str) else v)
                 for k, v in raw.items()}
            # join filter
            if by == "row_field" and key_field is not None:
                if str(_col_get(r, key_field)) != str(join_value):
                    continue
            r = _normalize_csv_row(r)
            # keep_where / drop_where
            if flt.get("keep_where") and not eval_predicate(r, flt["keep_where"]):
                continue
            if flt.get("drop_where") and eval_predicate(r, flt["drop_where"]):
                continue
            seg = _row_to_segment(r, time_spec, text_spec, extras_spec)
            segments.append(seg)
    return segments


def _row_to_segment(
    row: dict[str, Any], time_spec: dict[str, Any], text_spec: dict[str, Any], extras_spec: list[str]
) -> dict[str, Any]:
    units = time_spec.get("units", "s")
    start = end = point = None
    if time_spec.get("point") is not None:
        point = to_seconds(_dotted_get(row, str(time_spec["point"])), units)
    if time_spec.get("start") is not None:
        start = to_seconds(_dotted_get(row, str(time_spec["start"])), units)
    if time_spec.get("end") is not None:
        end = to_seconds(_dotted_get(row, str(time_spec["end"])), units)
    text = compose_text(row, text_spec)
    return _segment(start=start, end=end, point=point, text=text,
                    extras=_gather_extras(row, extras_spec))


def read_anno_json_by_key(
    payload: Any, channel: dict[str, Any], episode_id: str, tok: dict[str, str], duration_s: float | None
) -> list[dict[str, Any]]:
    """Navigate select_path, apply filter, compose_text, kind point/interval.

    Handles three select_path shapes seen in the recipes:
      * 'annotations.{take_uid}'                 (ego-exo4d: nested under 'annotations')
      * 'annotations.{episode}_ego.mp4[.Fine-level]'  (egome: keyed by mp4 name)
      * '[video_name].events[label==X]'          (holoassist: list of take dicts)
    """
    join = channel.get("join", {})
    select_path = join.get("select_path", "")
    kind = channel.get("kind", "interval")
    time_spec = channel.get("time", {})
    text_spec = channel.get("text", {})
    extras_spec = channel.get("extras", [])
    flt = channel.get("filter", {})
    key_field = join.get("key_field")
    key_in_file = join.get("key_in_file")

    # Resolve the join key value.
    key_val = episode_id
    if key_field == "take_uid":
        key_val = tok.get("take_uid", episode_id)
    if key_in_file:
        key_val = _fill(key_in_file, {**tok, "episode_id": episode_id})

    items = _navigate_select(payload, select_path, episode_id, key_val, tok)

    segments: list[dict[str, Any]] = []
    for it in _iter_records(items):
        if not isinstance(it, dict):
            continue
        if flt.get("keep_where") and not eval_predicate(it, flt["keep_where"]):
            continue
        if flt.get("drop_where") and eval_predicate(it, flt["drop_where"]):
            continue
        seg = _json_record_to_segment(it, kind, time_spec, text_spec, extras_spec, duration_s)
        segments.append(seg)
    return segments


def _navigate_select(payload: Any, select_path: str, episode_id: str, key_val: str, tok: dict[str, str]) -> Any:
    """Resolve a recipe select_path to the raw record container for this episode."""
    sp = select_path

    # holoassist: '[video_name].events[label==...]' over a top-level list of take dicts.
    if sp.startswith("[") and isinstance(payload, list):
        take = next((x for x in payload
                     if isinstance(x, dict) and x.get("video_name") == episode_id), None)
        if take is None:
            return []
        m = re.search(r"\.events\b", sp)
        if m:
            return take.get("events", [])
        return take

    # ego-exo4d / egome: dotted path under a dict; fill {take_uid}/{episode_id} placeholders.
    filled = _fill(sp, {**tok, "episode_id": episode_id, "take_uid": tok.get("take_uid", "")})
    # The filled key may itself contain dots that are literal (mp4 names, Fine-level).
    # Strategy: peel known prefixes step by step.
    cur = payload
    # 'annotations.<KEY>[.<SUBKEY>...]'
    parts = filled.split(".")
    # Greedy: try to match the longest contiguous existing keys.
    i = 0
    while i < len(parts) and isinstance(cur, dict):
        # try progressively longer joins to handle keys that contain dots
        matched = False
        for j in range(len(parts), i, -1):
            cand = ".".join(parts[i:j])
            if cand in cur:
                cur = cur[cand]
                i = j
                matched = True
                break
        if not matched:
            return []
    return cur


def _iter_records(items: Any):
    """Flatten ego-exo4d's pass layer; pass through plain lists; wrap singletons.

    ego-exo4d atomic: items = list of passes, each with 'descriptions'[]; flatten to
    description records but keep the parent pass's 'rejected' so drop_where works.
    ego-exo4d expert: items = list of entries, each with 'commentary_data'[]; flatten,
    propagating entry-level 'task_name'.
    """
    if isinstance(items, list):
        for it in items:
            if isinstance(it, dict) and "descriptions" in it:
                rejected = it.get("rejected")
                for d in it.get("descriptions", []):
                    if isinstance(d, dict):
                        yield {**d, "rejected": rejected}
            elif isinstance(it, dict) and "commentary_data" in it:
                task_name = it.get("task_name")
                for c in it.get("commentary_data", []):
                    if isinstance(c, dict):
                        yield {**c, "task_name": task_name}
            else:
                yield it
    elif isinstance(items, dict):
        yield items


def _json_record_to_segment(
    rec: dict[str, Any],
    kind: str,
    time_spec: dict[str, Any],
    text_spec: dict[str, Any],
    extras_spec: list[str],
    duration_s: float | None,
) -> dict[str, Any]:
    units = time_spec.get("units", "s")
    start = end = point = None
    if time_spec.get("whole_episode"):
        start, end = 0.0, (duration_s if duration_s is not None else None)
    elif kind == "point":
        point = to_seconds(_path_get(rec, str(time_spec.get("point"))), units)
    else:
        start = to_seconds(_path_get(rec, str(time_spec.get("start"))), units)
        if time_spec.get("end") is not None:
            end = to_seconds(_path_get(rec, str(time_spec.get("end"))), units)
        elif time_spec.get("end_from") == "start+duration_approx":
            dur = _num(_path_get(rec, "duration_approx"))
            end = (start + dur) if (start is not None and dur is not None) else None
    text = compose_text(rec, text_spec)
    return _segment(start=start, end=end, point=point, text=text,
                    extras=_gather_extras(rec, extras_spec))


def read_anno_pandas_pickle(
    local: Path,
    channel: dict[str, Any],
    episode_id: str,
    puller: Puller,
    root: str,
) -> list[dict[str, Any]]:
    """hd-epic narration: pandas.read_pickle; filter rows by video_id==episode;
    apply erratum denylist csv; build interval segments with main_actions etc as extras."""
    import pandas as pd

    df = pd.read_pickle(local)
    join = channel.get("join", {})
    key_field = join.get("key_field", "video_id")
    df = df[df[key_field].astype(str) == str(episode_id)]

    # Erratum denylist.
    flt = channel.get("filter", {})
    if flt.get("denylist_file"):
        deny_rel = f"{root}/{flt['denylist_file']}".replace("//", "/")
        deny_local = puller.pull(deny_rel)
        deny = pd.read_csv(deny_local)
        deny_field = flt.get("denylist_field")
        if deny_field and deny_field in deny.columns:
            deny_ids = set(deny[deny_field].astype(str))
            id_cols = [c for c in df.columns if c.lower().endswith("narration_id")]
            id_col = "unique_narration_id" if "unique_narration_id" in df.columns else (id_cols[0] if id_cols else None)
            if id_col:
                df = df[~df[id_col].astype(str).isin(deny_ids)]

    time_spec = channel.get("time", {})
    text_spec = channel.get("text", {})
    extras_spec = channel.get("extras", [])
    units = time_spec.get("units", "s")
    segments = []
    for _, row in df.iterrows():
        r = row.to_dict()
        start = to_seconds(r.get(time_spec.get("start")), units)
        end = to_seconds(r.get(time_spec.get("end")), units)
        text = compose_text(r, text_spec)
        extras = {}
        for key in extras_spec:
            v = r.get(key)
            # list-valued cells (main_actions) -> json-safe list
            try:
                import numpy as _np
                if isinstance(v, _np.ndarray):
                    v = v.tolist()
            except Exception:
                pass
            extras[key] = v
        segments.append(_segment(start=start, end=end, text=text, extras=extras))
    return segments


# =========================================================================== #
# GAZE PROJECTION
# =========================================================================== #
def project_gaze(
    gaze: GazeTable,
    video: VideoMeta | None,
    *,
    puller: Puller,
    root: str,
    tok: dict[str, str],
    n_samples: int = 10,
) -> dict[str, Any]:
    """Project up to ``n_samples`` gaze samples to mp4 pixels per projection.method.

    Returns {'method', 'samples': [{t_s, x_px, y_px, in_frame}...], 'notes': [...]}.
    Non-destructive: does not mutate the gaze rows (projection is a derived view).
    """
    proj = (gaze.projection or {})
    method = proj.get("method", "none")
    w = video.width if video else None
    h = video.height if video else None
    notes: list[str] = []

    rows = [r for r in gaze.rows if r.get("valid", True)]
    if not rows:
        rows = gaze.rows
    # Pick evenly spaced indices.
    idxs = _even_indices(len(rows), n_samples)

    if method == "already_2d":
        out = []
        for i in idxs:
            r = rows[i]
            x, y = r.get("x"), r.get("y")
            out.append(_pix(r.get("t_s"), x, y, w, h))
        return {"method": method, "samples": out, "notes": notes}

    if method == "normalize_by_dims":
        fd = (gaze.projection or {}).get("frame_dims") if gaze.projection else None
        # frame_dims may live on gaze_format, not projection; caller passes via gaze.notes
        fdims = _frame_dims_for(gaze, w, h)
        out = []
        for i in idxs:
            r = rows[i]
            x, y = r.get("x"), r.get("y")
            if x is None or y is None:
                out.append(_pix(r.get("t_s"), None, None, w, h))
                continue
            if gaze.coordinate_space == "normalized_2d":
                px = x * (w or fdims[0])
                py = y * (h or fdims[1])
            else:  # pixel_2d already in source dims -> rescale to mp4 dims
                sx = (w / fdims[0]) if (w and fdims[0]) else 1.0
                sy = (h / fdims[1]) if (h and fdims[1]) else 1.0
                px, py = x * sx, y * sy
            out.append(_pix(r.get("t_s"), px, py, w, h))
        return {"method": method, "samples": out, "notes": notes}

    if method == "projectaria_cpf":
        return _project_aria(gaze, video, puller=puller, root=root, tok=tok, idxs=idxs, notes=notes)

    if method == "psi_pinhole_ray":
        return _project_psi(gaze, video, puller=puller, root=root, tok=tok, idxs=idxs, notes=notes)

    return {"method": method, "samples": [], "notes": ["no projection method"]}


def _frame_dims_for(gaze: GazeTable, w: int | None, h: int | None) -> tuple[float, float]:
    # frame_dims captured into projection dict by extract_episode
    fd = None
    if gaze.projection:
        fd = gaze.projection.get("_frame_dims")
    if fd and len(fd) == 2:
        return float(fd[0]), float(fd[1])
    return float(w or 1), float(h or 1)


def _even_indices(n: int, k: int) -> list[int]:
    if n == 0:
        return []
    if n <= k:
        return list(range(n))
    return [round(i * (n - 1) / (k - 1)) for i in range(k)]


def _pix(t_s, x, y, w, h) -> dict[str, Any]:
    in_frame = None
    if x is not None and y is not None and w and h:
        in_frame = (0 <= x <= w) and (0 <= y <= h)
    return {
        "t_s": t_s,
        "x_px": round(x, 3) if x is not None else None,
        "y_px": round(y, 3) if y is not None else None,
        "in_frame": in_frame,
    }


# --- Aria CPF projection (validated logic, reused verbatim) ---------------- #
_NATIVE = {
    "camera-rgb": (2880, 2880),
    "camera-slam-left": (640, 480),
    "camera-slam-right": (640, 480),
}


def _build_aria_device(line: dict[str, Any]):
    from projectaria_tools.core.calibration import (
        CameraCalibration, CameraModelType, DeviceCalibration,
        DeviceCadExtrinsics, DeviceVersion,
    )
    from projectaria_tools.core.sophus import SE3
    import numpy as np

    def build_cam(c):
        t = np.array(c["T_Device_Camera"]["Translation"])
        qw, (qx, qy, qz) = c["T_Device_Camera"]["UnitQuaternion"]
        T = SE3.from_quat_and_translation(qw, np.array([qx, qy, qz]), t)
        w, h = _NATIVE[c["Label"]]
        return CameraCalibration(
            c["Label"], CameraModelType.FISHEYE624,
            np.array(c["Projection"]["Params"]), T, w, h, None, np.pi, c["SerialNumber"],
        )

    cams = {c["Label"]: build_cam(c) for c in line["CameraCalibrations"]}
    cad = DeviceCadExtrinsics(DeviceVersion.Gen1, "DVT-S", "camera-slam-left")
    dc = DeviceCalibration(cams, {}, {}, {}, {}, cad, "DVT-S", "camera-slam-left", DeviceVersion.Gen1)
    return dc, cams["camera-rgb"]


def _load_calib_lines(local: Path) -> list[dict[str, Any]]:
    lines = []
    with open(local, encoding="utf-8", errors="replace") as fh:
        for ln in fh:
            ln = ln.strip()
            if ln:
                try:
                    lines.append(json.loads(ln))
                except json.JSONDecodeError:
                    pass
    return lines


def _deref_slam_index(puller: Puller, root: str, tok: dict[str, str], notes: list[str]) -> str | None:
    """hd-epic: SLAM/multi/vrs_to_multi_slam.json maps 'P##/<takeid>.vrs' -> index N."""
    part = tok.get("participant", "")
    ep = tok.get("episode_id", "")
    map_rel = f"{root}/SLAM-and-Gaze/{part}/SLAM/multi/vrs_to_multi_slam.json".replace("//", "/")
    try:
        local = puller.pull(map_rel)
        mapping = json.loads(Path(local).read_text())
    except Exception as e:  # noqa
        notes.append(f"vrs_to_multi_slam.json load failed: {e}")
        return None
    want = f"{part}/{ep}.vrs"
    if want in mapping:
        notes.append(f"slam_index({want}) = {mapping[want]}")
        return str(mapping[want])
    # fall back: match on the take id alone
    for k, v in mapping.items():
        if ep in k:
            notes.append(f"slam_index(~{ep}) = {v}")
            return str(v)
    notes.append(f"slam_index: {want} not in vrs_to_multi_slam.json")
    return None


def _nearest_calib(calib_lines: list[dict[str, Any]], target_us: float) -> dict[str, Any]:
    best, bestd = calib_lines[0], None
    for cl in calib_lines:
        ts = cl.get("tracking_timestamp_us")
        if ts is None:
            continue
        d = abs(ts - target_us)
        if bestd is None or d < bestd:
            best, bestd = cl, d
    return best


def _project_aria(gaze, video, *, puller, root, tok, idxs, notes):
    import numpy as np
    from projectaria_tools.core.mps import EyeGaze
    from projectaria_tools.core.mps.utils import get_gaze_vector_reprojection

    proj = gaze.projection or {}
    calib_spec = (proj.get("calibration") or {})
    calib_tmpl = calib_spec.get("file", "")
    local_tok = {**tok, "episode_id": tok.get("episode_id", "")}
    # hd-epic: {slam_index} must be dereferenced from vrs_to_multi_slam.json.
    if "{slam_index}" in calib_tmpl:
        idx = _deref_slam_index(puller, root, local_tok, notes)
        if idx is None:
            return {"method": "projectaria_cpf", "samples": [], "notes": notes + ["slam_index deref failed"]}
        local_tok["slam_index"] = idx
    calib_rel = _fill(calib_tmpl, local_tok)
    calib_rel = f"{root}/{calib_rel}".replace("//", "/")
    try:
        calib_local = puller.pull(calib_rel)
    except Exception as e:  # noqa
        return {"method": "projectaria_cpf", "samples": [], "notes": [f"calib pull failed: {e}"]}
    calib_lines = _load_calib_lines(calib_local)
    if not calib_lines:
        return {"method": "projectaria_cpf", "samples": [], "notes": ["empty online_calibration.jsonl"]}

    mp4_w = video.width if video else 2880
    scale = (mp4_w / 2880.0) if mp4_w else 1.0
    notes.append(f"scale = mp4_w/2880 = {scale:.5f}; make_upright=True; per-sample depth_m")

    samples = []
    # gaze rows carry left_yaw/right_yaw/pitch/depth + the raw tracking ts.
    for i in idxs:
        r = gaze.rows[i]
        ts_us = r.get("_tracking_timestamp_us")
        cl = _nearest_calib(calib_lines, ts_us) if ts_us is not None else calib_lines[0]
        dc, rgb_cam = _build_aria_device(cl)
        ly = r.get("left_yaw")
        ry = r.get("right_yaw")
        pitch = r.get("pitch")
        depth = r.get("depth") or 1.0
        if ly is None or ry is None or pitch is None:
            samples.append(_pix(r.get("t_s"), None, None, mp4_w, video.height if video else None))
            continue
        eg = EyeGaze()
        eg.yaw = 0.5 * (ly + ry)
        eg.pitch = pitch
        eg.depth = depth
        px = get_gaze_vector_reprojection(eg, "camera-rgb", dc, rgb_cam, depth_m=depth, make_upright=True)
        if px is None:
            samples.append({"t_s": r.get("t_s"), "x_px": None, "y_px": None, "in_frame": False})
            continue
        px = np.asarray(px) * scale
        samples.append(_pix(r.get("t_s"), float(px[0]), float(px[1]),
                            mp4_w, video.height if video else None))
    return {"method": "projectaria_cpf", "samples": samples, "notes": notes}


# --- psi pinhole ray projection (holoassist) ------------------------------- #
def _parse_psi_intrinsics(local: Path) -> dict[str, Any]:
    """psi HoloLensCaptureExporter Intrinsics.txt:
       24 floats = [3x3 M(9), distortion(0..7=8?), then FL, FLx, FLy, PPx, PPy, ?, W, H].
       We confirmed: indices 0,4=FLx,FLy; 2,5=PPx,PPy; last two = W,H.
       Radial/tangential distortion sit in the middle block (all 0 on this take)."""
    vals = [float(x) for x in re.split(r"[\t ]+", local.read_text().strip())]
    M = vals[0:9]
    flx, fly = M[0], M[4]
    ppx, ppy = M[2], M[5]
    w = int(round(vals[-2]))
    h = int(round(vals[-1]))
    distortion = vals[9:17]  # Brown-Conrady block (k1,k2,p1,p2,k3,... typically)
    return {"flx": flx, "fly": fly, "ppx": ppx, "ppy": ppy, "w": w, "h": h,
            "distortion": distortion, "raw": vals}


def _parse_pose_sync(local: Path) -> list[dict[str, Any]]:
    """Pose_sync.txt rows: col0 rel_time_s, col1 FILETIME, cols2..17 = 4x4 row-major."""
    import numpy as np
    out = []
    with open(local, encoding="utf-8", errors="replace") as fh:
        for ln in fh:
            parts = ln.strip().split("\t")
            if len(parts) < 18:
                continue
            t = float(parts[0])
            mat = np.array([float(x) for x in parts[2:18]], dtype=float).reshape(4, 4)
            out.append({"t_s": t, "T": mat})
    return out


def _nearest_pose(poses: list[dict[str, Any]], t_s: float):
    best, bestd = None, None
    for p in poses:
        d = abs(p["t_s"] - t_s)
        if bestd is None or d < bestd:
            best, bestd = p, d
    return best


def _project_psi(gaze, video, *, puller, root, tok, idxs, notes):
    import numpy as np

    proj = gaze.projection or {}
    calib_spec = (proj.get("calibration") or {})
    extr_spec = (proj.get("extrinsics") or {})
    intr_rel = f"{root}/{_fill(calib_spec.get('file',''), tok)}".replace("//", "/")
    pose_rel = f"{root}/{_fill(extr_spec.get('file',''), tok)}".replace("//", "/")
    try:
        intr_local = puller.pull(intr_rel)
        pose_local = puller.pull(pose_rel)
    except Exception as e:  # noqa
        return {"method": "psi_pinhole_ray", "samples": [], "notes": [f"pull failed: {e}"]}

    intr = _parse_psi_intrinsics(intr_local)
    poses = _parse_pose_sync(pose_local)
    notes.append(f"intrinsics W={intr['w']} H={intr['h']} FLx={intr['flx']:.1f} FLy={intr['fly']:.1f} "
                 f"PPx={intr['ppx']:.1f} PPy={intr['ppy']:.1f}; distortion={['%.4g'%d for d in intr['distortion']]}")
    notes.append("pinhole projection P+t*V -> camera frame via inv(Pose 4x4) -> intrinsics; "
                 "Brown-Conrady distortion is 0 on this take (refinement noted)")
    mp4_w = video.width if video else intr["w"]
    # Intrinsics W==mp4 W (896) -> no rescale per recipe.
    rescale = (mp4_w / intr["w"]) if intr["w"] else 1.0

    samples = []
    for i in idxs:
        r = gaze.rows[i]
        t = r.get("t_s")
        P = np.array([r.get("px"), r.get("py"), r.get("pz")], dtype=float)
        V = np.array([r.get("vx"), r.get("vy"), r.get("vz")], dtype=float)
        if t is None or np.any(np.isnan(P)) or np.any(np.isnan(V)):
            samples.append(_pix(t, None, None, mp4_w, video.height if video else None))
            continue
        pose = _nearest_pose(poses, t)
        if pose is None:
            samples.append(_pix(t, None, None, mp4_w, video.height if video else None))
            continue
        T = pose["T"]  # camera->world (rig pose): translation ~= gaze origin P (world)
        R = T[:3, :3]
        tr = T[:3, 3]
        # Gaze point 1 m along the ray, expressed in the camera (PV) frame.
        pt_world = P + 1.0 * V
        pt_cam = R.T @ (pt_world - tr)
        X, Y, Z = pt_cam
        if X == 0:
            samples.append(_pix(t, None, None, mp4_w, video.height if video else None))
            continue
        # psi/HoloLens MathNet camera convention: forward = +X, image_x left = +Y,
        # image_y up = +Z. Pinhole: u = ppx - flx*(Y/X); v = ppy - fly*(Z/X).
        # (Validated on R0027-12-GoPro: 9/10 samples land tightly around the
        #  principal point, the 1 miss is an untracked sample.) Brown-Conrady
        #  distortion is 0 on this take -> omitted; add as a refinement when nonzero.
        if X <= 0:  # gaze pointing behind the camera -> not visible
            samples.append({"t_s": t, "x_px": None, "y_px": None, "in_frame": False})
            continue
        u = intr["ppx"] - intr["flx"] * (Y / X)
        v = intr["ppy"] - intr["fly"] * (Z / X)
        u *= rescale
        v *= rescale
        samples.append(_pix(t, float(u), float(v), mp4_w, video.height if video else None))
    return {"method": "psi_pinhole_ray", "samples": samples, "notes": notes}


# =========================================================================== #
# EPISODE ASSEMBLY
# =========================================================================== #
def extract_episode(
    slug: str,
    episode_id: str,
    puller: Puller,
    *,
    recipes_dir: str | Path | None = None,
    sample_extra: dict[str, Any] | None = None,
    keep_mp4: bool = False,
) -> EpisodeBundle:
    """Resolve, pull, and parse one episode into an EpisodeBundle.

    Enforces the emit policy (video + gaze + >=1 annotation channel => emitted).
    Large pulled mp4s are deleted after ffprobe unless ``keep_mp4`` is True.
    """
    recipe = load_recipe(slug, recipes_dir)
    root = recipe["root"]
    extra = dict(sample_extra or {})
    tok = _episode_tokens(slug, episode_id, extra)

    bundle = EpisodeBundle(dataset=slug, episode_id=episode_id)
    warnings = bundle.warnings

    # --- bridge take_name <-> take_uid (ego-exo4d) and metadata resolution --- #
    takes_index: list[dict[str, Any]] | None = None
    take_entry: dict[str, Any] | None = None
    if recipe.get("episode", {}).get("bridge"):
        bridge = recipe["episode"]["bridge"]
        bridge_rel = f"{root}/{bridge['file']}".replace("//", "/")
        try:
            takes_local = puller.pull(bridge_rel)
            takes_index = json.loads(Path(takes_local).read_text())
            kf = bridge.get("key_field", "take_name")
            vf = bridge.get("value_field", "take_uid")
            for t in takes_index:
                if t.get(kf) == episode_id:
                    take_entry = t
                    tok["take_uid"] = t.get(vf, "")
                    tok["take_name"] = episode_id
                    break
            if take_entry is None:
                warnings.append(f"bridge: episode {episode_id} not found in {bridge['file']}")
        except Exception as e:  # noqa
            warnings.append(f"bridge load failed: {e}")

    def metadata_resolver(spec: dict[str, Any]) -> str | None:
        # ego-exo4d video from takes.json frame_aligned_videos.*.rgb.relative_path
        if take_entry is None:
            return None
        field = spec.get("field", "")
        if "frame_aligned_videos" in field:
            fav = take_entry.get("frame_aligned_videos", {})
            for cam, v in fav.items():
                if isinstance(v, dict) and "rgb" in v:
                    rel = v["rgb"].get("relative_path")
                    if rel:
                        return f"takes/takes/{episode_id}/{rel}"
        return None

    # --- VIDEO --- #
    video: VideoMeta | None = None
    vsel = recipe["video"]["select"]
    vrel = resolve_file(puller, root, vsel, tok, metadata_resolver=metadata_resolver)
    pulled_mp4: Path | None = None
    if vrel is None:
        warnings.append(f"video not found (select={vsel.get('path_template') or vsel.get('glob')})")
    else:
        try:
            pulled_mp4 = puller.pull(vrel)
            video = ffprobe_video(pulled_mp4)
            video.path = vrel  # record source-relative path, not the temp copy
            # prefer shipped duration where the recipe says so
            video = _apply_shipped_video(recipe, root, puller, tok, episode_id, video, take_entry, warnings)
        except Exception as e:  # noqa
            warnings.append(f"video ffprobe failed: {e}")
    bundle.video = video

    # --- GAZE --- #
    gaze: GazeTable | None = None
    gsel = recipe["gaze"]["select"]
    gf = recipe["gaze"]["gaze_format"]
    grel = resolve_file(puller, root, gsel, tok)
    if grel is None:
        warnings.append(f"gaze not found (select={gsel.get('path_template') or gsel.get('prefer')})")
    else:
        try:
            glocal = puller.pull(grel)
            gaze = _parse_gaze(slug, episode_id, glocal, gf, video, grel)
            gaze.source_path = grel
            # stash frame_dims + raw tracking ts for projection
            if gaze.projection is not None and gf.get("frame_dims"):
                gaze.projection = dict(gaze.projection)
                gaze.projection["_frame_dims"] = gf["frame_dims"]
        except Exception as e:  # noqa
            import traceback
            warnings.append(f"gaze parse failed: {e}")
            traceback.print_exc()
    bundle.gaze = gaze

    # --- ANNOTATIONS --- #
    duration_s = video.duration_s if video else None
    for channel in recipe.get("annotations", []):
        name = channel["name"]
        try:
            ch = _parse_annotation(
                slug, episode_id, channel, puller, root, tok, duration_s,
                takes_index=takes_index, take_entry=take_entry,
            )
            if ch is None:
                warnings.append(f"annotation '{name}': source not found")
                continue
            if ch.segment_count == 0:
                warnings.append(f"annotation '{name}': 0 segments for this episode")
            bundle.annotations.append(ch)
        except Exception as e:  # noqa
            import traceback
            warnings.append(f"annotation '{name}' failed: {e}")
            traceback.print_exc()

    # --- emit policy --- #
    have_video = bundle.video is not None and bundle.video.duration_s
    have_gaze = bundle.gaze is not None and bundle.gaze.sample_count > 0
    have_anno = any(a.segment_count > 0 for a in bundle.annotations)
    reasons = []
    if not have_video:
        reasons.append("no video")
    if not have_gaze:
        reasons.append("no gaze")
    if not have_anno:
        reasons.append("no annotation channel with segments")
    bundle.emitted = not reasons
    bundle.emit_reason = None if bundle.emitted else "; ".join(reasons)

    # --- delete large pulled mp4s (only our own scp'd temp copies, never the
    #     in-place local_root/nfs source, which is read-only) --- #
    if pulled_mp4 is not None and not keep_mp4 and puller.owns(pulled_mp4):
        try:
            if pulled_mp4.exists() and pulled_mp4.stat().st_size > 5_000_000:
                pulled_mp4.unlink()
        except OSError:
            pass

    return bundle


def _apply_shipped_video(recipe, root, puller, tok, episode_id, video, take_entry, warnings):
    meta = recipe.get("metadata", {})
    ps = (meta.get("prefer_shipped") or {}).get("video")
    if not ps:
        return video
    fields = ps.get("fields", {})
    # ego-exo4d: duration from takes.json entry
    if take_entry is not None and "duration_s" in fields:
        src = fields["duration_s"]
        if src == "duration_sec" and take_entry.get("duration_sec"):
            ffd = video.duration_s
            video.duration_s = float(take_entry["duration_sec"])
            video.source = "shipped(takes.json):ffprobe-crosscheck"
            if ffd and abs(ffd - video.duration_s) > 1.0:
                warnings.append(f"video duration shipped={video.duration_s:.2f} vs ffprobe={ffd:.2f}")
    return video


def _parse_gaze(slug, episode_id, glocal, gf, video, grel) -> GazeTable:
    reader = gf["reader"]
    fps = video.fps if video else gf.get("time", {}).get("fps")
    if reader == "begaze_txt":
        fr = _egtea_frame_range(episode_id)
        gaze = read_gaze_begaze_txt(glocal, gf, fps, fr)
    elif reader == "csv":
        gaze = read_gaze_csv(glocal, gf, fps)
        _attach_aria_ts(slug, gaze, glocal, gf)
    elif reader == "npy":
        gaze = read_gaze_npy(glocal, gf, fps)
    elif reader == "whitespace_txt":
        gaze = read_gaze_whitespace_txt(glocal, gf, fps)
    else:
        raise ValueError(f"unknown gaze reader {reader!r}")
    return gaze


def _attach_aria_ts(slug, gaze, glocal, gf):
    """For projectaria_cpf datasets, keep the raw tracking_timestamp_us per row so the
    projector can pick the nearest online-calibration line."""
    proj = gf.get("projection", {})
    if proj.get("method") != "projectaria_cpf":
        return
    import csv as _csv
    src = gf.get("time", {}).get("source")
    with open(glocal, newline="", encoding="utf-8", errors="replace") as fh:
        for i, r in enumerate(_csv.DictReader(fh)):
            if i >= len(gaze.rows):
                break
            v = r.get(str(src))
            gaze.rows[i]["_tracking_timestamp_us"] = float(v) if v else None


def _egtea_frame_range(episode_id: str) -> tuple[int, int] | None:
    m = re.search(r"-F(\d+)-F(\d+)$", episode_id)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def _parse_annotation(
    slug, episode_id, channel, puller, root, tok, duration_s, *, takes_index, take_entry
) -> AnnotationChannel | None:
    reader = channel["reader"]
    name = channel["name"]
    kind = channel["kind"]
    sel = channel["select"]

    if reader == "pandas_pickle":
        rel = resolve_file(puller, root, sel, tok)
        if rel is None:
            return None
        local = puller.pull(rel)
        segs = read_anno_pandas_pickle(local, channel, episode_id, puller, root)
        ch = annotation_stats(name, kind, segs, duration_s)
        ch.source_path = rel
        return ch

    if reader == "csv":
        rel = resolve_file(puller, root, sel, tok)
        if rel is None:
            return None
        local = puller.pull(rel)
        # egtea joins on the clip prefix = episode_id minus trailing -F<start>-F<end>.
        join_value = episode_id
        if slug == "egtea":
            join_value = re.sub(r"-F\d+-F\d+$", "", episode_id)
        segs = read_anno_csv(local, channel, episode_id, tok, join_value=join_value)
        ch = annotation_stats(name, kind, segs, duration_s)
        ch.source_path = rel
        return ch

    if reader == "json_by_key":
        rel = resolve_file(puller, root, sel, tok)
        if rel is None:
            return None
        local = puller.pull(rel)
        payload = json.loads(Path(local).read_text(encoding="utf-8", errors="replace"))
        segs = read_anno_json_by_key(payload, channel, episode_id, tok, duration_s)
        ch = annotation_stats(name, kind, segs, duration_s)
        ch.source_path = rel
        return ch

    raise ValueError(f"unknown annotation reader {reader!r}")
