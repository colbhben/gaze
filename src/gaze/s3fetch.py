"""Raw-S3 fetch layer for the viewer (no /nfs mount required).

Shared by the eval-cache viewer (gaze-67t.3.3) and the processed-manifest-from-S3 viewer
(gaze-67t.3.4). Small files (json/jsonl/parquet) are downloaded once and cached on disk;
video is streamed via HTTP range GETs (boto3 ``get_object(Range=...)``), never fully
downloaded, so the viewer scrubs over large clips without pulling them whole.

boto3 is imported lazily so the core ``gaze`` package keeps zero hard dependencies — only
the S3 viewer paths need it (``pip install gaze[s3]``).
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable, Iterator

from .s3 import split_s3_uri

_CLIENT = None


def _client():
    global _CLIENT
    if _CLIENT is None:
        try:
            import boto3  # lazy: only required for the S3 viewer paths
        except ImportError as exc:  # pragma: no cover - import-guard
            raise RuntimeError(
                "boto3 is required for S3 viewer sources. Install it with: pip install 'gaze[s3]'"
            ) from exc
        _CLIENT = boto3.client("s3")
    return _CLIENT


def cache_dir(root: str | Path | None = None) -> Path:
    return Path(root or ".gaze-cache") / "s3fetch"


def _cache_path(uri: str, root: str | Path | None) -> Path:
    bucket, key = split_s3_uri(uri)
    digest = hashlib.sha1(f"{bucket}/{key}".encode("utf-8")).hexdigest()[:16]
    return cache_dir(root) / bucket / f"{digest}-{Path(key).name}"


def get_bytes(uri: str, cache_root: str | Path | None = None, use_cache: bool = True) -> bytes:
    """Fetch an object's bytes, caching small files on disk by default."""
    if use_cache:
        path = _cache_path(uri, cache_root)
        if path.exists():
            return path.read_bytes()
    bucket, key = split_s3_uri(uri)
    body = _client().get_object(Bucket=bucket, Key=key)["Body"].read()
    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    return body


def get_text(uri: str, cache_root: str | Path | None = None, use_cache: bool = True) -> str:
    return get_bytes(uri, cache_root=cache_root, use_cache=use_cache).decode("utf-8")


def object_size(uri: str) -> int:
    bucket, key = split_s3_uri(uri)
    return int(_client().head_object(Bucket=bucket, Key=key)["ContentLength"])


def read_range(uri: str, start: int, length: int, chunk: int = 1024 * 1024) -> Iterator[bytes]:
    """Yield ``length`` bytes of ``uri`` starting at ``start`` via a ranged GET."""
    if length <= 0:
        return
    bucket, key = split_s3_uri(uri)
    end = start + length - 1
    body = _client().get_object(Bucket=bucket, Key=key, Range=f"bytes={start}-{end}")["Body"]
    remaining = length
    for piece in body.iter_chunks(chunk_size=chunk):
        if not piece:
            break
        if len(piece) > remaining:
            piece = piece[:remaining]
        yield piece
        remaining -= len(piece)
        if remaining <= 0:
            break


class S3VideoHandle:
    """A range-capable video object backed by S3 (mirrors the local-file video path)."""

    def __init__(self, uri: str):
        self.uri = uri
        self._size: int | None = None

    @property
    def size(self) -> int:
        if self._size is None:
            self._size = object_size(self.uri)
        return self._size

    def read_range(self, start: int, length: int) -> Iterator[bytes]:
        return read_range(self.uri, start, length)


class LocalVideoHandle:
    """A range-capable video object backed by a local file."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.size = self.path.stat().st_size

    def read_range(self, start: int, length: int, chunk: int = 1024 * 1024) -> Iterator[bytes]:
        with self.path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining > 0:
                piece = handle.read(min(chunk, remaining))
                if not piece:
                    break
                yield piece
                remaining -= len(piece)
