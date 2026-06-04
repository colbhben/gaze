from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any


MODALITY_ASSET_KEYS: dict[str, tuple[str, ...]] = {
    "video": ("video_main_rgb", "main_vrs", "takes", "video"),
    "gaze": ("mps_eye_gaze", "take_eye_gaze", "eyes", "gaze"),
    "annotation": (
        "annotations",
        "annotation",
        "ground_truth",
        "narration_atomic_action_csv",
        "narration_activity_summarization_csv",
        "narration_motion_narration_csv",
    ),
    "depth": ("ahat_depth", "take_point_cloud", "mps_slam_points", "semidense_observations"),
    "pose": ("hand_data", "mps_slam_trajectories"),
}


DATASET_ALIASES = {
    "aea": "aea",
    "ariaeverydayactivities": "aea",
    "aria_everyday_activities": "aea",
    "hot3d": "hot3d",
    "hot3daria": "hot3d",
    "nymeria": "nymeria",
    "holoassist": "holoassist",
    "egtea": "egtea",
    "egtea gaze+": "egtea",
    "ego-exo4d": "ego-exo4d",
    "ego4d": "ego-exo4d",
}


@dataclass
class DatasetDoc:
    name: str
    slug: str
    instructions: list[str] = field(default_factory=list)
    urls: dict[str, list[str]] = field(default_factory=dict)
    manifest_paths: list[Path] = field(default_factory=list)
    credential_paths: list[Path] = field(default_factory=list)


@dataclass
class Asset:
    dataset: str
    sequence_id: str
    asset_key: str
    modality: str
    filename: str
    url: str | None
    size_bytes: int | None = None
    sha1: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetCatalog:
    root: Path
    docs: dict[str, DatasetDoc]

    def dataset_names(self) -> list[str]:
        return sorted(self.docs)

    def manifest_assets(self, dataset: str | None = None) -> list[Asset]:
        assets: list[Asset] = []
        for slug, doc in self.docs.items():
            if dataset and normalize_dataset_name(dataset) != slug:
                continue
            for manifest in doc.manifest_paths:
                assets.extend(load_manifest_assets(manifest, slug))
            assets.extend(load_provider_manifest_assets(doc))
            assets.extend(load_direct_doc_assets(doc))
        return assets


def normalize_dataset_name(name: str) -> str:
    compact = re.sub(r"[^a-z0-9+_-]+", "", name.lower())
    spaced = name.strip().lower()
    return DATASET_ALIASES.get(spaced, DATASET_ALIASES.get(compact, compact))


def load_catalog(root: str | Path = ".") -> DatasetCatalog:
    repo_root = Path(root)
    docs = parse_datasets_md(repo_root / "DATASETS.md")
    for doc in docs.values():
        doc.manifest_paths = [path for path in doc.manifest_paths if path.exists()]
    return DatasetCatalog(root=repo_root, docs=docs)


def parse_datasets_md(path: str | Path) -> dict[str, DatasetDoc]:
    source = Path(path)
    docs: dict[str, DatasetDoc] = {}
    current: DatasetDoc | None = None
    current_section: str | None = None
    for raw_line in source.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("# "):
            name = line[2:].strip()
            slug = normalize_dataset_name(name)
            current = DatasetDoc(name=name, slug=slug)
            docs[slug] = current
            current_section = None
            continue
        if current is None:
            continue
        if line.startswith("## "):
            current_section = line[3:].strip().lower()
            current.urls.setdefault(current_section, [])
            continue
        if line.endswith(".json") and not line.startswith("http"):
            current.manifest_paths.append((source.parent / line).resolve())
            continue
        if line.endswith(".txt") and not line.startswith("http"):
            current.credential_paths.append((source.parent / line).resolve())
            continue
        if line.startswith("http://") or line.startswith("https://"):
            if current_section == "instructions":
                current.instructions.append(line)
            else:
                section = current_section or "links"
                current.urls.setdefault(section, []).append(line)
    return docs


def load_manifest_assets(path: str | Path, dataset_slug: str) -> list[Asset]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    sequences = data.get("sequences", {})
    assets: list[Asset] = []
    for sequence_id, sequence_assets in sequences.items():
        for key, info in sequence_assets.items():
            if not isinstance(info, dict):
                continue
            url = info.get("download_url")
            filename = info.get("filename") or Path(url or key).name
            modality = modality_for_asset_key(key)
            assets.append(
                Asset(
                    dataset=dataset_slug,
                    sequence_id=sequence_id,
                    asset_key=key,
                    modality=modality,
                    filename=filename,
                    url=url,
                    size_bytes=info.get("file_size_bytes"),
                    sha1=info.get("sha1sum"),
                )
            )
    return assets


def load_provider_manifest_assets(doc: DatasetDoc) -> list[Asset]:
    if doc.slug != "ego-exo4d":
        return []
    credential_path = str(doc.credential_paths[0]) if doc.credential_paths else "download_links/ego-exo4d.txt"
    release = "v2"
    base = "s3://ego4d-consortium-sharing/egoexo-public"
    common = {
        "download_kind": "egoexo_manifest",
        "release": release,
        "credential_path": credential_path,
    }
    return [
        Asset(
            dataset=doc.slug,
            sequence_id=release,
            asset_key="takes",
            modality="video",
            filename="takes",
            url=f"{base}/{release}/takes/manifest.json",
            extra={**common, "views": ["ego"], "benchmarks": ["atomic_action_descriptions", "expert_commentary"]},
        ),
        Asset(
            dataset=doc.slug,
            sequence_id=release,
            asset_key="take_eye_gaze",
            modality="gaze",
            filename="take_eye_gaze",
            url=f"{base}/{release}/take_eye_gaze/manifest.json",
            extra={**common, "benchmarks": ["atomic_action_descriptions", "expert_commentary"]},
        ),
        Asset(
            dataset=doc.slug,
            sequence_id=release,
            asset_key="annotations_atomic_descriptions",
            modality="annotation",
            filename="annotations_atomic_descriptions",
            url=f"{base}/{release}/annotations/manifest.json",
            extra={**common, "benchmarks": ["atomic_descriptions"]},
        ),
        Asset(
            dataset=doc.slug,
            sequence_id=release,
            asset_key="annotations_expert_commentary",
            modality="annotation",
            filename="annotations_expert_commentary",
            url=f"{base}/{release}/annotations/manifest.json",
            extra={**common, "benchmarks": ["expert_commentary"]},
        ),
    ]


def load_direct_doc_assets(doc: DatasetDoc) -> list[Asset]:
    assets: list[Asset] = []
    for section, urls in doc.urls.items():
        modality = normalize_modality(section)
        for index, url in enumerate(urls):
            assets.append(
                Asset(
                    dataset=doc.slug,
                    sequence_id="dataset",
                    asset_key=section,
                    modality=modality,
                    filename=Path(url.split("?", 1)[0]).name or f"{section}-{index}",
                    url=url,
                )
            )
    return assets


def normalize_modality(name: str) -> str:
    lowered = name.lower()
    if lowered in {"annotations", "annotation", "text annotation", "text annotations"}:
        return "annotation"
    if lowered == "depth":
        return "depth"
    if lowered == "pose":
        return "pose"
    if lowered == "video":
        return "video"
    if lowered == "gaze":
        return "gaze"
    return "other"


def modality_for_asset_key(key: str) -> str:
    for modality, keys in MODALITY_ASSET_KEYS.items():
        if key in keys:
            return modality
    return "other"


def filter_assets(
    assets: list[Asset],
    datasets: set[str] | None = None,
    modalities: set[str] | None = None,
    sequences: set[str] | None = None,
) -> list[Asset]:
    normalized_datasets = {normalize_dataset_name(item) for item in datasets or set()}
    normalized_modalities = {normalize_modality(item) if item not in MODALITY_ASSET_KEYS else item for item in modalities or set()}
    result = []
    for asset in assets:
        if normalized_datasets and asset.dataset not in normalized_datasets:
            continue
        if normalized_modalities and asset.modality not in normalized_modalities:
            continue
        if sequences and asset.sequence_id not in sequences:
            continue
        result.append(asset)
    return result


def summarize_assets(assets: list[Asset]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for asset in assets:
        key = (asset.dataset, asset.modality)
        row = grouped.setdefault(
            key,
            {
                "dataset": asset.dataset,
                "modality": asset.modality,
                "assets": 0,
                "sequences": set(),
                "bytes": 0,
                "unknown_bytes": 0,
            },
        )
        row["assets"] += 1
        row["sequences"].add(asset.sequence_id)
        if asset.size_bytes is None:
            row["unknown_bytes"] += 1
        else:
            row["bytes"] += asset.size_bytes
    rows = []
    for row in grouped.values():
        row = dict(row)
        row["sequences"] = len(row["sequences"])
        row["gb"] = round(row["bytes"] / 1_000_000_000, 3)
        rows.append(row)
    return sorted(rows, key=lambda item: (item["dataset"], item["modality"]))
