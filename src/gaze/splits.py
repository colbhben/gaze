from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any

from .table import read_table


@dataclass
class SplitRequest:
    name: str = "default"
    ratios: dict[str, float] | None = None
    seed: int = 0
    mode: str = "heterogeneous"
    include_datasets: set[str] | None = None
    include_modalities: set[str] | None = None
    group_by: str = "dataset"
    stratify_by: str | None = None


def create_split(canonical_root: str | Path, request: SplitRequest) -> dict[str, Any]:
    root = Path(canonical_root)
    manifest_path = root / "manifest.parquet"
    episodes = read_table(manifest_path)
    selected = filter_manifest(episodes, request)
    ratios = normalize_ratios(request.ratios or {"train": 0.8, "holdout": 0.2})
    rng = random.Random(request.seed)
    split_ids = {name: [] for name in ratios}

    if request.mode == "homogeneous":
        groups = group_records(selected, request.group_by)
        for records in groups.values():
            assign_group(records, ratios, rng, split_ids)
    elif request.mode == "heterogeneous":
        assign_group(selected, ratios, rng, split_ids)
    else:
        raise ValueError("mode must be homogeneous or heterogeneous")

    result = {
        "name": request.name,
        "seed": request.seed,
        "mode": request.mode,
        "ratios": ratios,
        "filters": {
            "include_datasets": sorted(request.include_datasets or []),
            "include_modalities": sorted(request.include_modalities or []),
            "group_by": request.group_by,
            "stratify_by": request.stratify_by,
        },
        "splits": split_ids,
    }
    output = root / "splits" / f"{request.name}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def filter_manifest(records: list[dict[str, Any]], request: SplitRequest) -> list[dict[str, Any]]:
    result = []
    for record in records:
        if request.include_datasets and record.get("dataset") not in request.include_datasets:
            continue
        if request.include_modalities:
            modalities = set(str(record.get("modalities", "")).split(","))
            if not request.include_modalities.issubset(modalities):
                continue
        result.append(record)
    return result


def normalize_ratios(ratios: dict[str, float]) -> dict[str, float]:
    if not ratios:
        raise ValueError("ratios cannot be empty")
    total = sum(float(value) for value in ratios.values())
    if total <= 0:
        raise ValueError("ratios must sum to a positive value")
    return {key: float(value) / total for key, value in ratios.items()}


def group_records(records: list[dict[str, Any]], group_by: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        key = str(record.get(group_by, ""))
        groups.setdefault(key, []).append(record)
    return groups


def assign_group(records: list[dict[str, Any]], ratios: dict[str, float], rng: random.Random, split_ids: dict[str, list[str]]) -> None:
    shuffled = list(records)
    rng.shuffle(shuffled)
    boundaries = []
    running = 0.0
    for name, ratio in ratios.items():
        running += ratio
        boundaries.append((name, running))
    for index, record in enumerate(shuffled):
        fraction = (index + 0.5) / max(1, len(shuffled))
        for name, boundary in boundaries:
            if fraction <= boundary:
                split_ids[name].append(episode_id(record))
                break


def episode_id(record: dict[str, Any]) -> str:
    return f"{record.get('dataset')}:{record.get('episode_id')}"
