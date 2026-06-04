from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
import unittest

from gaze.server import GazeRequestHandler
from gaze.table import write_table


class ServerTests(unittest.TestCase):
    def test_video_supports_byte_ranges_and_annotation_intervals_api(self) -> None:
        root = self.make_canonical_root()
        handler = TestHandler({"Range": "bytes=2-5"})
        handler.send_file(root / "episodes" / "toy" / "ep1" / "video.mp4", "video/mp4")
        self.assertEqual(handler.status, 206)
        self.assertEqual(handler.response_headers["Accept-Ranges"], "bytes")
        self.assertEqual(handler.response_headers["Content-Range"], "bytes 2-5/10")
        self.assertEqual(handler.wfile.getvalue(), b"2345")

        handler = TestHandler()
        handler.canonical_root = root
        handler.handle_episode_get("/api/episodes/toy%3Aep1/annotation_intervals")
        payload = json.loads(handler.wfile.getvalue())
        self.assertEqual(payload["rows"], [{"start_s": 0.0, "end_s": 1.0, "label": "a", "text": "step a"}])

    def make_canonical_root(self) -> Path:
        import tempfile

        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        episode_root = root / "episodes" / "toy" / "ep1"
        episode_root.mkdir(parents=True)
        (episode_root / "video.mp4").write_bytes(b"0123456789")
        write_table(
            [{"start_s": 0.0, "end_s": 1.0, "label": "a", "text": "step a"}],
            episode_root / "annotation_intervals.parquet",
        )
        write_table(
            [{"dataset": "toy", "episode_id": "ep1", "duration_s": 1.0, "episode_path": "episodes/toy/ep1", "modalities": "annotations,video"}],
            root / "manifest.parquet",
        )
        (episode_root / "episode.json").write_text(
            json.dumps(
                {
                    "dataset": "toy",
                    "episode_id": "ep1",
                    "duration_s": 1.0,
                    "files": {"video": "video.mp4", "annotation_intervals": "annotation_intervals.parquet"},
                    "modalities": ["annotations", "video"],
                }
            ),
            encoding="utf-8",
        )
        return root

    def tearDown(self) -> None:
        tmp = getattr(self, "tmp", None)
        if tmp:
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()


class TestHandler(GazeRequestHandler):
    canonical_root: Path

    def __init__(self, headers: dict[str, str] | None = None) -> None:
        self.headers = headers or {}
        self.wfile = BytesIO()
        self.status = None
        self.response_headers = {}

    def send_response(self, code: int, message: str | None = None) -> None:
        self.status = code

    def send_header(self, keyword: str, value: str) -> None:
        self.response_headers[keyword] = value

    def end_headers(self) -> None:
        return

    def send_error(self, code: int, message: str | None = None, explain: str | None = None) -> None:
        self.status = code
