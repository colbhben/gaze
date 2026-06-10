#!/usr/bin/env python3
"""Validate dataset curation recipes against the schema and cross-field rules.

Stdlib only (no jsonschema dependency), so it runs anywhere the gaze CLI does.
It performs a focused structural validation driven by recipe.schema.json
(types, required keys, enums, additionalProperties, anyOf-style key groups) plus
recipe-specific cross-field rules that a plain JSON Schema cannot express, e.g.:

  - dataset slug matches the filename and a DATASETS.md slug
  - annotation channel names are unique within a dataset
  - interval channels define start/end time; point channels define point
  - pixel_2d gaze declares frame_dims (needed to normalize)
  - projection.feasible=true carries a method != "none"
  - {placeholders} used in path templates are drawn from a known set

Usage:
  python scripts/validate_recipes.py            # validate all recipes/*.json
  python scripts/validate_recipes.py egtea      # validate one (slug or path)
Exit code 0 if all pass, 1 otherwise.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RECIPES_DIR = REPO_ROOT / "recipes"
SCHEMA_PATH = RECIPES_DIR / "recipe.schema.json"
DATASETS_MD = REPO_ROOT / "DATASETS.md"

KNOWN_PLACEHOLDERS = {
    "episode_id", "take_name", "take_uid", "take", "prefix", "video_uid",
    "session", "participant", "slam_index", "key",
}


class Issue:
    __slots__ = ("path", "message")

    def __init__(self, path: str, message: str) -> None:
        self.path = path
        self.message = message

    def __str__(self) -> str:
        return f"  [{self.path}] {self.message}"


# --------------------------------------------------------------------------- #
# Minimal JSON-Schema structural validator (subset used by recipe.schema.json)
# --------------------------------------------------------------------------- #
class SchemaValidator:
    def __init__(self, schema: dict) -> None:
        self.schema = schema
        self.defs = schema.get("$defs", {})

    def validate(self, instance) -> list[Issue]:
        issues: list[Issue] = []
        self._check(instance, self.schema, "$", issues)
        return issues

    def _resolve(self, schema: dict) -> dict:
        ref = schema.get("$ref")
        if not ref:
            return schema
        assert ref.startswith("#/$defs/"), f"unsupported $ref: {ref}"
        return self.defs[ref.split("/")[-1]]

    def _check(self, value, schema: dict, path: str, issues: list[Issue]) -> None:
        schema = self._resolve(schema)

        if "enum" in schema and value not in schema["enum"]:
            issues.append(Issue(path, f"{value!r} not in {schema['enum']}"))
            return
        if "const" in schema and value != schema["const"]:
            issues.append(Issue(path, f"{value!r} != const {schema['const']!r}"))

        types = schema.get("type")
        if types is not None:
            if not self._type_ok(value, types):
                issues.append(Issue(path, f"expected type {types}, got {type(value).__name__}"))
                return

        if isinstance(value, str) and "pattern" in schema:
            if not re.search(schema["pattern"], value):
                issues.append(Issue(path, f"{value!r} does not match /{schema['pattern']}/"))
        if isinstance(value, str) and schema.get("minLength") and len(value) < schema["minLength"]:
            issues.append(Issue(path, "string too short"))

        if isinstance(value, dict):
            self._check_object(value, schema, path, issues)
        elif isinstance(value, list):
            self._check_array(value, schema, path, issues)

        # Combinators
        for branch in schema.get("allOf", []):
            self._check_conditional(value, branch, path, issues)
        if "anyOf" in schema:
            self._check_anyof(value, schema["anyOf"], path, issues)

    def _check_conditional(self, value, branch: dict, path: str, issues: list[Issue]) -> None:
        # supports {if, then} and plain subschemas inside allOf
        if "if" in branch:
            if isinstance(value, dict) and self._matches(value, branch["if"]):
                self._check(value, branch["then"], path, issues)
        else:
            self._check(value, branch, path, issues)

    def _matches(self, value, cond: dict) -> bool:
        # shallow match used by if-clauses (properties const + required)
        for key, sub in cond.get("properties", {}).items():
            if "const" in sub and value.get(key) != sub["const"]:
                return False
        for key in cond.get("required", []):
            if key not in value:
                return False
        return True

    def _check_anyof(self, value, branches: list[dict], path: str, issues: list[Issue]) -> None:
        # anyOf in this schema is only used as "at least one of these required-key sets"
        for branch in branches:
            sub_issues: list[Issue] = []
            self._check(value, branch, path, sub_issues)
            if not sub_issues:
                return
        required_sets = [b.get("required") for b in branches if b.get("required")]
        if required_sets:
            issues.append(Issue(path, f"must satisfy one of required key-sets {required_sets}"))
        else:
            issues.append(Issue(path, "does not satisfy any anyOf branch"))

    def _check_object(self, value: dict, schema: dict, path: str, issues: list[Issue]) -> None:
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                issues.append(Issue(path, f"missing required key '{key}'"))
        if schema.get("additionalProperties") is False:
            for key in value:
                if key not in props:
                    issues.append(Issue(path, f"unexpected key '{key}'"))
        ap = schema.get("additionalProperties")
        for key, sub_value in value.items():
            if key in props:
                self._check(sub_value, props[key], f"{path}.{key}", issues)
            elif isinstance(ap, dict):
                self._check(sub_value, ap, f"{path}.{key}", issues)

    def _check_array(self, value: list, schema: dict, path: str, issues: list[Issue]) -> None:
        if schema.get("minItems") and len(value) < schema["minItems"]:
            issues.append(Issue(path, f"expected >= {schema['minItems']} items"))
        if schema.get("maxItems") and len(value) > schema["maxItems"]:
            issues.append(Issue(path, f"expected <= {schema['maxItems']} items"))
        item_schema = schema.get("items")
        if item_schema:
            for index, item in enumerate(value):
                self._check(item, item_schema, f"{path}[{index}]", issues)

    @staticmethod
    def _type_ok(value, types) -> bool:
        if isinstance(types, str):
            types = [types]
        ok = False
        for type_name in types:
            if type_name == "object":
                ok = ok or isinstance(value, dict)
            elif type_name == "array":
                ok = ok or isinstance(value, list)
            elif type_name == "string":
                ok = ok or isinstance(value, str)
            elif type_name == "integer":
                ok = ok or (isinstance(value, int) and not isinstance(value, bool))
            elif type_name == "number":
                ok = ok or (isinstance(value, (int, float)) and not isinstance(value, bool))
            elif type_name == "boolean":
                ok = ok or isinstance(value, bool)
            elif type_name == "null":
                ok = ok or value is None
        return ok


# --------------------------------------------------------------------------- #
# Cross-field rules (beyond what the structural schema enforces)
# --------------------------------------------------------------------------- #
def datasets_md_slugs() -> set[str]:
    """Resolve DATASETS.md headings to catalog slugs using the same normalizer
    the catalog uses, so e.g. 'EGTEA Gaze+' -> 'egtea' matches the recipe."""
    if not DATASETS_MD.exists():
        return set()
    try:
        sys.path.insert(0, str(REPO_ROOT / "src"))
        from gaze.datasets import normalize_dataset_name
    except Exception:
        normalize_dataset_name = lambda name: re.sub(r"[^a-z0-9+_-]+", "", name.lower())
    slugs = set()
    for line in DATASETS_MD.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            slugs.add(normalize_dataset_name(line[2:].strip()))
    return slugs


def iter_placeholders(text: str):
    return re.findall(r"\{([a-z_]+)\}", text)


def check_cross_field(recipe: dict, filename: str, md_slugs: set[str]) -> list[Issue]:
    issues: list[Issue] = []
    slug = recipe.get("dataset", "")

    if filename and slug and filename != slug:
        issues.append(Issue("$.dataset", f"slug {slug!r} != filename stem {filename!r}"))
    if md_slugs and slug and slug not in md_slugs:
        issues.append(Issue("$.dataset", f"slug {slug!r} not found as a DATASETS.md heading (known: {sorted(md_slugs)})"))

    # annotation channel names unique
    names = [ch.get("name") for ch in recipe.get("annotations", [])]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        issues.append(Issue("$.annotations", f"duplicate channel names: {sorted(dupes)}"))

    # per-channel: interval needs start|end; point needs point
    for index, ch in enumerate(recipe.get("annotations", [])):
        cpath = f"$.annotations[{index}]({ch.get('name')})"
        time = ch.get("time", {})
        if ch.get("kind") == "point" and "point" not in time:
            issues.append(Issue(cpath, "kind=point requires time.point"))
        if ch.get("kind") == "interval" and not ("start" in time or time.get("whole_episode")):
            issues.append(Issue(cpath, "kind=interval requires time.start (or time.whole_episode=true)"))
        if "start" in time and "end" not in time and "end_from" not in time and not time.get("whole_episode"):
            issues.append(Issue(cpath, "interval has start but no end/end_from (set end, end_from, or whole_episode)"))
        join = ch.get("join", {})
        if join.get("by") == "row_field" and not join.get("key_field"):
            issues.append(Issue(cpath, "join.by=row_field requires join.key_field"))
        if join.get("by") == "key" and not (join.get("key_field") or join.get("select_path")):
            issues.append(Issue(cpath, "join.by=key requires key_field or select_path"))

    # gaze format rules
    gaze = recipe.get("gaze", {}).get("gaze_format", {})
    space = gaze.get("coordinate_space")
    if space == "pixel_2d" and "frame_dims" not in gaze and not gaze.get("frame_dims_unknown"):
        issues.append(Issue("$.gaze.gaze_format", "coordinate_space=pixel_2d needs frame_dims (or frame_dims_unknown=true)"))
    proj = gaze.get("projection", {})
    if proj.get("feasible") is True and proj.get("method") in (None, "none"):
        issues.append(Issue("$.gaze.gaze_format.projection", "feasible=true but method is none"))
    if proj.get("feasible") is False and proj.get("method") not in (None, "none"):
        issues.append(Issue("$.gaze.gaze_format.projection", "feasible=false but a concrete method is set"))
    # frame_index time requires fps
    gtime = gaze.get("time", {})
    if gtime.get("units") == "frame_index" and "fps" not in gtime:
        issues.append(Issue("$.gaze.gaze_format.time", "units=frame_index requires fps"))
    for variant in gaze.get("variants", []):
        vt = variant.get("time", {})
        if vt.get("units") in ("frame_index", "timecode_hmsf") and "fps" not in vt:
            issues.append(Issue("$.gaze.gaze_format.variants", f"variant {variant.get('version')} time units need fps"))

    # placeholder hygiene across all path templates / globs
    for tmpl_path, text in _iter_templates(recipe):
        for ph in iter_placeholders(text):
            if ph not in KNOWN_PLACEHOLDERS:
                issues.append(Issue(tmpl_path, f"unknown placeholder {{{ph}}} (known: {sorted(KNOWN_PLACEHOLDERS)})"))

    return issues


def _iter_templates(node, path: str = "$"):
    """Yield (path, string) for keys that hold path templates/globs."""
    template_keys = {"path_template", "glob", "file", "denylist_file"}
    if isinstance(node, dict):
        for key, value in node.items():
            child = f"{path}.{key}"
            if key in template_keys and isinstance(value, str):
                yield child, value
            elif key == "needs" and isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, str):
                        yield f"{child}[{i}]", item
            else:
                yield from _iter_templates(value, child)
    elif isinstance(node, list):
        for index, item in enumerate(node):
            yield from _iter_templates(item, f"{path}[{index}]")


# --------------------------------------------------------------------------- #
def validate_file(path: Path, validator: SchemaValidator, md_slugs: set[str]) -> list[Issue]:
    try:
        recipe = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [Issue("$", f"invalid JSON: {exc}")]
    issues = validator.validate(recipe)
    issues += check_cross_field(recipe, path.stem, md_slugs)
    return issues


def main(argv: list[str]) -> int:
    if not SCHEMA_PATH.exists():
        print(f"error: schema not found at {SCHEMA_PATH}", file=sys.stderr)
        return 1
    validator = SchemaValidator(json.loads(SCHEMA_PATH.read_text(encoding="utf-8")))
    md_slugs = datasets_md_slugs()

    if argv:
        targets = []
        for arg in argv:
            candidate = Path(arg)
            if not candidate.exists():
                candidate = RECIPES_DIR / f"{arg}.json"
            targets.append(candidate)
    else:
        targets = sorted(p for p in RECIPES_DIR.glob("*.json") if p.name != "recipe.schema.json" and not p.name.startswith("_"))

    total_issues = 0
    for path in targets:
        if not path.exists():
            print(f"MISSING {path}")
            total_issues += 1
            continue
        issues = validate_file(path, validator, md_slugs)
        if issues:
            print(f"FAIL {path.name} ({len(issues)} issue(s))")
            for issue in issues:
                print(issue)
            total_issues += len(issues)
        else:
            print(f"ok   {path.name}")

    print()
    if total_issues:
        print(f"{total_issues} issue(s) across {len(targets)} recipe(s)")
        return 1
    print(f"all {len(targets)} recipe(s) valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
