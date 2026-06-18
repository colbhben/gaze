"""Episode sources for the gaze viewer.

The viewer's HTTP handler talks to a ``Source``, which abstracts where episodes come from:

  * ``LocalSource``        — a local canonical root (manifest.parquet + episodes/<dataset>/<id>/);
                             the original behavior, unchanged.
  * ``S3CanonicalSource``  — the SAME canonical layout rooted at an s3:// URI, fetched on demand
                             via boto3 with no /nfs mount (bead gaze-67t.3.4).
  * ``S3EvalSource``       — an eval cache (summary.json + results.jsonl) produced by the gaze
                             eval pipeline; renders GT vs predicted gaze + per-episode metrics
                             (bead gaze-67t.3.3).

Small files are cached on disk; video is range-streamed (see :mod:`gaze.s3fetch`). boto3 is only
imported by the S3 sources, so the local path keeps zero dependencies.
"""
from __future__ import annotations

import json
from os.path import basename
from pathlib import Path
from typing import Any

from .table import read_table


class Source:
    """Interface the request handler depends on."""

    def episodes(self) -> list[dict]:
        raise NotImplementedError

    def episode_doc(self, dataset: str, episode_id: str) -> dict | None:
        raise NotImplementedError

    def table_rows(self, dataset: str, episode_id: str, key: str) -> list[dict]:
        raise NotImplementedError

    def open_video(self, dataset: str, episode_id: str):
        raise NotImplementedError

    def describe(self) -> str:
        return self.__class__.__name__


class LocalSource(Source):
    """Local canonical root: manifest.parquet + episodes/<dataset>/<episode_id>/."""

    def __init__(self, canonical_root: str | Path):
        self.canonical_root = Path(canonical_root).resolve()

    def _episode_root(self, dataset: str, episode_id: str) -> Path:
        return self.canonical_root / "episodes" / dataset / episode_id

    def episodes(self) -> list[dict]:
        manifest = read_table(self.canonical_root / "manifest.parquet")
        return [{"id": f"{row.get('dataset')}:{row.get('episode_id')}", **row} for row in manifest]

    def episode_doc(self, dataset: str, episode_id: str) -> dict | None:
        doc_path = self._episode_root(dataset, episode_id) / "episode.json"
        if not doc_path.exists():
            return None
        return json.loads(doc_path.read_text(encoding="utf-8"))

    def table_rows(self, dataset: str, episode_id: str, key: str) -> list[dict]:
        doc = self.episode_doc(dataset, episode_id) or {}
        table = doc.get("files", {}).get(key)
        if not table:
            return []
        return read_table(self._episode_root(dataset, episode_id) / table)

    def open_video(self, dataset: str, episode_id: str):
        from .s3fetch import LocalVideoHandle

        doc = self.episode_doc(dataset, episode_id) or {}
        video = doc.get("files", {}).get("video")
        if not video:
            return None
        path = self._episode_root(dataset, episode_id) / video
        if not path.exists():
            return None
        return LocalVideoHandle(path)

    def describe(self) -> str:
        return str(self.canonical_root)


class S3CanonicalSource(Source):
    """The canonical viewer layout rooted at an s3:// URI (no /nfs). Bead gaze-67t.3.4."""

    def __init__(self, root_uri: str, cache_root: str | Path | None = None):
        self.root_uri = root_uri.rstrip("/")
        self.cache_root = cache_root

    def _episode_uri(self, dataset: str, episode_id: str) -> str:
        return f"{self.root_uri}/episodes/{dataset}/{episode_id}"

    def episodes(self) -> list[dict]:
        from . import s3fetch

        # Prefer the JSONL export (no parquet dep); fall back to manifest.parquet.jsonl.
        rows: list[dict] = []
        for name in ("manifest.jsonl", "manifest.parquet.jsonl"):
            try:
                # The manifest is a mutable index -> read uncached so a re-export isn't masked.
                text = s3fetch.get_text(f"{self.root_uri}/{name}", use_cache=False)
            except Exception:
                continue
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
            break
        return [{"id": f"{row.get('dataset')}:{row.get('episode_id')}", **row} for row in rows]

    def episode_doc(self, dataset: str, episode_id: str) -> dict | None:
        from . import s3fetch

        try:
            text = s3fetch.get_text(f"{self._episode_uri(dataset, episode_id)}/episode.json", cache_root=self.cache_root)
        except Exception:
            return None
        return json.loads(text)

    def table_rows(self, dataset: str, episode_id: str, key: str) -> list[dict]:
        from . import s3fetch

        doc = self.episode_doc(dataset, episode_id) or {}
        table = doc.get("files", {}).get(key)
        if not table:
            return []
        uri = f"{self._episode_uri(dataset, episode_id)}/{table}"
        local = s3fetch._cache_path(uri, self.cache_root)
        if not local.exists():
            data = s3fetch.get_bytes(uri, cache_root=self.cache_root)
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(data)
        return read_table(local)

    def open_video(self, dataset: str, episode_id: str):
        from .s3fetch import S3VideoHandle

        doc = self.episode_doc(dataset, episode_id) or {}
        video = doc.get("files", {}).get("video")
        if not video:
            return None
        return S3VideoHandle(f"{self._episode_uri(dataset, episode_id)}/{video}")

    def describe(self) -> str:
        return self.root_uri


class Molmo2ManifestSource(Source):
    """A molmo2 training manifest (``manifest.jsonl`` + flat ``videos/<slug>/<id>__segK.mp4``).

    Unlike the canonical layout (``episodes/<dataset>/<id>/episode.json`` + per-modality
    parquet), the training manifest is one JSONL row per CLIP, with per-frame ``points``
    (raw ``{x,y}`` pixels on a ``resolution``-square frame) and a relative ``video``
    pointer. This adapter exposes each clip as a viewer "episode" so ``gaze serve`` can
    render the rectified clips + per-frame gaze natively, local OR over s3://.

    Each row ``id`` is ``dataset:episode#segK``; the handler splits a viewer episode id
    on the FIRST ``:`` into ``(dataset, rest)``, so we index rows by ``(dataset, rest)``
    where ``rest`` is everything after the first colon (``episode#segK``).
    """

    def __init__(self, root: str | Path, cache_root: str | Path | None = None):
        from .s3 import is_s3_uri

        self.is_s3 = is_s3_uri(root)
        self.root = str(root).rstrip("/") if self.is_s3 else Path(root).resolve()
        self.cache_root = cache_root
        self._rows: dict[tuple[str, str], dict] | None = None

    # -- manifest load (uncached on S3: a re-export shouldn't be masked) -------- #
    def _manifest_text(self) -> str:
        if self.is_s3:
            from . import s3fetch
            return s3fetch.get_text(f"{self.root}/manifest.jsonl", use_cache=False)
        return (self.root / "manifest.jsonl").read_text(encoding="utf-8")

    def _load(self) -> dict[tuple[str, str], dict]:
        if self._rows is not None:
            return self._rows
        rows: dict[tuple[str, str], dict] = {}
        for line in self._manifest_text().splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rid = row.get("id") or f"{row.get('dataset')}:{row.get('episode_id')}#seg{row.get('seg_index')}"
            dataset, _, rest = rid.partition(":")
            rows[(dataset, rest)] = row
        self._rows = rows
        return rows

    def episodes(self) -> list[dict]:
        out = []
        for (dataset, rest), row in self._load().items():
            meta = row.get("metadata", {})
            out.append({
                "id": f"{dataset}:{rest}",
                "dataset": dataset,
                "episode_id": rest,
                "duration_s": (meta.get("clip_end_time") or 0) - (meta.get("clip_start_time") or 0),
                "modalities": "video,gaze",
                "num_frames": row.get("num_frames"),
            })
        out.sort(key=lambda e: e["id"])
        return out

    def episode_doc(self, dataset: str, episode_id: str) -> dict | None:
        row = self._load().get((dataset, episode_id))
        if row is None:
            return None
        meta = row.get("metadata", {})
        return {
            "dataset": dataset,
            "episode_id": episode_id,
            "duration_s": (meta.get("clip_end_time") or 0) - (meta.get("clip_start_time") or 0),
            "resolution": row.get("resolution"),
            "fps": row.get("fps"),
            "label": meta.get("final_annotation") or meta.get("annotation_text") or "",
            "files": {"video": row.get("video")},
            "modalities": ["video", "gaze"],
        }

    def table_rows(self, dataset: str, episode_id: str, key: str) -> list[dict]:
        row = self._load().get((dataset, episode_id))
        if row is None:
            return []
        if key == "gaze":
            # One viewer gaze row per frame: timestamps[j] + points[j] (raw px on the
            # resolution-square frame; the frontend divides by frameSide=resolution).
            ts = row.get("timestamps") or []
            pts = row.get("points") or []
            out = []
            for j, frame_pts in enumerate(pts):
                t = ts[j] if j < len(ts) else j / float(row.get("fps") or 1.0)
                if frame_pts:  # non-empty -> [{x,y}]; empty -> masked frame, no dot
                    p = frame_pts[0]
                    out.append({"time_s": float(t), "x_px": float(p["x"]), "y_px": float(p["y"])})
                else:
                    out.append({"time_s": float(t), "x_px": None, "y_px": None})
            return out
        if key in ("annotations", "annotation_intervals"):
            meta = row.get("metadata", {})
            label = meta.get("final_annotation") or meta.get("annotation_text") or ""
            if not label:
                return []
            dur = (meta.get("clip_end_time") or 0) - (meta.get("clip_start_time") or 0)
            return [{"start_s": 0.0, "end_s": float(dur), "role": "final", "text": label}]
        return []

    def open_video(self, dataset: str, episode_id: str):
        row = self._load().get((dataset, episode_id))
        if row is None or not row.get("video"):
            return None
        if self.is_s3:
            from .s3fetch import S3VideoHandle
            return S3VideoHandle(f"{self.root}/{row['video']}")
        from .s3fetch import LocalVideoHandle
        path = self.root / row["video"]
        return LocalVideoHandle(path) if path.exists() else None

    def describe(self) -> str:
        return f"molmo2 manifest {self.root}"


class S3EvalSource(Source):
    """An eval cache (summary.json + results.jsonl) -> GT vs predicted gaze. Bead gaze-67t.3.3."""

    def __init__(self, cache_uri: str, cache_root: str | Path | None = None):
        self.cache_uri = cache_uri.rstrip("/")
        self.cache_root = cache_root
        self._records: dict[str, dict] | None = None
        self._summary: dict | None = None

    def _load(self) -> None:
        if self._records is not None:
            return
        from . import s3fetch

        # summary.json + results.jsonl are mutable index files (an eval can be re-run at the
        # same URI), so read them uncached -- caching by URI would serve a stale prior run.
        try:
            self._summary = json.loads(s3fetch.get_text(f"{self.cache_uri}/summary.json", use_cache=False))
        except Exception:
            self._summary = {}
        text = s3fetch.get_text(f"{self.cache_uri}/results.jsonl", use_cache=False)
        records: dict[str, dict] = {}
        for line in text.splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            dataset = rec.get("dataset") or "gaze"
            example_id = rec.get("example_id") or ""
            # episode_id keeps the full example id (which itself may contain ":"); the handler
            # splits on the FIRST ":" so dataset is recovered and episode_id is the remainder.
            records[f"{dataset}:{example_id}"] = rec
        self._records = records

    def _key(self, dataset: str, episode_id: str) -> str:
        return f"{dataset}:{episode_id}"

    @staticmethod
    def _triplets_to_rows(triplets: list[Any]) -> list[dict]:
        rows = []
        for trip in triplets or []:
            t, x, y = trip[0], trip[1], trip[2]
            rows.append({"time_s": float(t), "x_px": float(x), "y_px": float(y)})
        return rows

    def episodes(self) -> list[dict]:
        self._load()
        out = []
        for key, rec in self._records.items():
            m = rec.get("metrics", {})
            out.append({
                "id": key,
                "dataset": rec.get("dataset"),
                "episode_id": rec.get("example_id"),
                "modalities": "video,gaze,gaze_pred",
                "l2": m.get("l2"),
            })
        return out

    def episode_doc(self, dataset: str, episode_id: str) -> dict | None:
        self._load()
        rec = self._records.get(self._key(dataset, episode_id))
        if rec is None:
            return None
        duration = rec.get("video_duration") or rec.get("clip_end_time") or 0
        return {
            "dataset": dataset,
            "episode_id": episode_id,
            "duration_s": duration,
            "resolution": rec.get("frame_side"),
            "eval_mode": True,
            "label": rec.get("label", ""),
            "metrics": rec.get("metrics", {}),
            "prediction_text": rec.get("prediction_text", ""),
            "files": {"video": rec.get("video_s3_uri")},
            "modalities": ["video", "gaze", "gaze_pred"],
        }

    def table_rows(self, dataset: str, episode_id: str, key: str) -> list[dict]:
        self._load()
        rec = self._records.get(self._key(dataset, episode_id))
        if rec is None:
            return []
        if key == "gaze":
            return self._triplets_to_rows(rec.get("gt_triplets"))
        if key == "gaze_pred":
            return self._triplets_to_rows(rec.get("pred_triplets"))
        if key in ("annotations", "annotation_intervals"):
            label = rec.get("label") or ""
            if not label:
                return []
            duration = rec.get("video_duration") or rec.get("clip_end_time") or 0
            return [{"start_s": 0.0, "end_s": float(duration), "role": "final", "text": label}]
        return []

    def open_video(self, dataset: str, episode_id: str):
        from .s3fetch import S3VideoHandle

        self._load()
        rec = self._records.get(self._key(dataset, episode_id))
        if rec is None or not rec.get("video_s3_uri"):
            return None
        return S3VideoHandle(rec["video_s3_uri"])

    def describe(self) -> str:
        return f"eval cache {self.cache_uri}"
