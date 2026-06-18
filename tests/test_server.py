from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
import unittest

from gaze.server import GazeRequestHandler
from gaze.source import LocalSource, Molmo2ManifestSource, S3CanonicalSource, S3EvalSource
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
        handler.source = LocalSource(root)
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


class Molmo2ManifestSourceTests(unittest.TestCase):
    """Local molmo2 training manifest -> viewer episodes (clips + per-frame gaze)."""

    def setUp(self) -> None:
        import tempfile

        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        (root / "videos" / "nymeria").mkdir(parents=True)
        (root / "videos" / "nymeria" / "ep1__seg0.mp4").write_bytes(b"0123456789")
        row = {
            "id": "nymeria:ep1#seg0", "dataset": "nymeria", "episode_id": "ep1", "seg_index": 0,
            "video": "videos/nymeria/ep1__seg0.mp4", "fps": 2.0, "resolution": 378, "num_frames": 3,
            "timestamps": [0.0, 0.5, 1.0],
            "points": [[{"x": 100.0, "y": 200.0}], [], [{"x": 110.0, "y": 210.0}]],
            "metadata": {"clip_start_time": 30.0, "clip_end_time": 31.5,
                         "annotation_text": "walk", "final_annotation": "1) walk"},
        }
        (root / "manifest.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        self.root = root
        self.source = Molmo2ManifestSource(root)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_episodes_keyed_by_full_id(self) -> None:
        eps = self.source.episodes()
        self.assertEqual(len(eps), 1)
        self.assertEqual(eps[0]["id"], "nymeria:ep1#seg0")
        self.assertEqual(eps[0]["episode_id"], "ep1#seg0")
        self.assertAlmostEqual(eps[0]["duration_s"], 1.5)

    def test_episode_doc_and_video(self) -> None:
        # the handler splits on the FIRST ":" -> dataset="nymeria", episode_id="ep1#seg0"
        doc = self.source.episode_doc("nymeria", "ep1#seg0")
        self.assertEqual(doc["resolution"], 378)
        self.assertEqual(doc["files"]["video"], "videos/nymeria/ep1__seg0.mp4")
        vh = self.source.open_video("nymeria", "ep1#seg0")
        self.assertIsNotNone(vh)
        self.assertEqual(vh.size, 10)

    def test_gaze_rows_per_frame_with_masked_frame(self) -> None:
        rows = self.source.table_rows("nymeria", "ep1#seg0", "gaze")
        self.assertEqual(len(rows), 3)
        self.assertEqual((rows[0]["x_px"], rows[0]["y_px"]), (100.0, 200.0))
        self.assertIsNone(rows[1]["x_px"])  # empty points -> masked frame
        self.assertEqual(rows[2]["x_px"], 110.0)

    def test_annotation_interval_from_final(self) -> None:
        anno = self.source.table_rows("nymeria", "ep1#seg0", "annotation_intervals")
        self.assertEqual(anno, [{"start_s": 0.0, "end_s": 1.5, "role": "final", "text": "1) walk"}])

    def test_missing_episode_returns_none(self) -> None:
        self.assertIsNone(self.source.episode_doc("nymeria", "nope#seg9"))


class S3EvalSourceTests(unittest.TestCase):
    """S3EvalSource with a stub fetcher (no network, no boto3)."""

    def setUp(self) -> None:
        import gaze.s3fetch as s3fetch

        self._orig_get_text = s3fetch.get_text
        summary = {"schema_version": 1, "metrics": {"l2": 6.0}}
        results = [
            {
                "example_id": "ego-exo4d:clip_a#seg0",
                "dataset": "ego-exo4d",
                "label": "doing a thing",
                "frame_side": 378.0,
                "video_duration": 3.0,
                "clip_start_time": 0.0,
                "clip_end_time": 3.0,
                "video_s3_uri": "s3://bucket/videos/ego-exo4d/clip_a__seg0.mp4",
                "gt_triplets": [[0.0, 100.0, 200.0]],
                "pred_triplets": [[0.0, 110.0, 190.0]],
                "prediction_text": "<points coords=...>",
                "metrics": {"l2": 2.5, "acc@5": 1.0, "acc@10": 1.0, "acc@15": 1.0, "valid": 1.0, "n_gt": 1, "n_pred": 1},
            }
        ]
        self._blobs = {
            "s3://bucket/eval/summary.json": json.dumps(summary),
            "s3://bucket/eval/results.jsonl": "\n".join(json.dumps(r) for r in results),
        }

        def fake_get_text(uri, cache_root=None, use_cache=True):
            return self._blobs[uri]

        s3fetch.get_text = fake_get_text

    def tearDown(self) -> None:
        import gaze.s3fetch as s3fetch

        s3fetch.get_text = self._orig_get_text

    def test_eval_source_exposes_gt_pred_and_metrics(self) -> None:
        src = S3EvalSource("s3://bucket/eval/")
        episodes = src.episodes()
        self.assertEqual(len(episodes), 1)
        ep = episodes[0]
        # The id splits on the FIRST ":" -> dataset + (example_id may itself contain ":").
        dataset, episode_id = ep["id"].split(":", 1)
        self.assertEqual(dataset, "ego-exo4d")

        doc = src.episode_doc(dataset, episode_id)
        self.assertTrue(doc["eval_mode"])
        self.assertEqual(doc["resolution"], 378.0)
        self.assertEqual(doc["metrics"]["l2"], 2.5)
        self.assertEqual(doc["files"]["video"], "s3://bucket/videos/ego-exo4d/clip_a__seg0.mp4")

        gt = src.table_rows(dataset, episode_id, "gaze")
        self.assertEqual(gt, [{"time_s": 0.0, "x_px": 100.0, "y_px": 200.0}])
        pred = src.table_rows(dataset, episode_id, "gaze_pred")
        self.assertEqual(pred, [{"time_s": 0.0, "x_px": 110.0, "y_px": 190.0}])
        intervals = src.table_rows(dataset, episode_id, "annotation_intervals")
        self.assertEqual(intervals, [{"start_s": 0.0, "end_s": 3.0, "role": "final", "text": "doing a thing"}])


class S3CanonicalSourceTests(unittest.TestCase):
    """S3CanonicalSource with a stub fetcher (no network, no boto3)."""

    def setUp(self) -> None:
        import gaze.s3fetch as s3fetch

        self._orig_get_text = s3fetch.get_text
        manifest = [
            {"dataset": "toy", "episode_id": "ep1", "duration_s": 1.0, "modalities": "gaze,video"},
        ]
        doc = {
            "dataset": "toy",
            "episode_id": "ep1",
            "duration_s": 1.0,
            "files": {"video": "video.mp4", "gaze": "gaze.jsonl"},
        }
        gaze = {"time_s": 0.0, "x_px": 50.0, "y_px": 60.0}
        self._blobs = {
            "s3://bucket/proc/manifest.jsonl": json.dumps(manifest[0]),
            "s3://bucket/proc/episodes/toy/ep1/episode.json": json.dumps(doc),
            "s3://bucket/proc/episodes/toy/ep1/gaze.jsonl": json.dumps(gaze),
        }

        self._orig_get_bytes = s3fetch.get_bytes

        def fake_get_text(uri, cache_root=None, use_cache=True):
            if uri not in self._blobs:
                raise KeyError(uri)
            return self._blobs[uri]

        def fake_get_bytes(uri, cache_root=None, use_cache=True):
            return fake_get_text(uri).encode("utf-8")

        s3fetch.get_text = fake_get_text
        s3fetch.get_bytes = fake_get_bytes

        import tempfile

        self._cache = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        import gaze.s3fetch as s3fetch

        s3fetch.get_text = self._orig_get_text
        s3fetch.get_bytes = self._orig_get_bytes
        self._cache.cleanup()

    def test_canonical_source_lists_and_reads_episode(self) -> None:
        src = S3CanonicalSource("s3://bucket/proc/", cache_root=self._cache.name)
        episodes = src.episodes()
        self.assertEqual(episodes, [{"id": "toy:ep1", "dataset": "toy", "episode_id": "ep1", "duration_s": 1.0, "modalities": "gaze,video"}])

        doc = src.episode_doc("toy", "ep1")
        self.assertEqual(doc["files"]["video"], "video.mp4")

        rows = src.table_rows("toy", "ep1", "gaze")
        self.assertEqual(rows, [{"time_s": 0.0, "x_px": 50.0, "y_px": 60.0}])

        handle = src.open_video("toy", "ep1")
        self.assertEqual(handle.uri, "s3://bucket/proc/episodes/toy/ep1/video.mp4")


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
