from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from gaze.config import default_config
from gaze.rectify import rectify_dataset
from gaze.splits import SplitRequest, create_split
from gaze.table import read_table, write_csv, write_table
from gaze.validate import validate_canonical_root


class RectifyValidateSplitTests(unittest.TestCase):
    def test_rectify_validate_and_split_synthetic_episode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_root = tmp_path / "raw"
            canonical_root = tmp_path / "canonical"
            make_raw_episode(raw_root, "toy", "ep1")
            make_raw_episode(raw_root, "toy", "ep2")

            manifest = rectify_dataset(raw_root, canonical_root, default_config())
            self.assertEqual(len(manifest), 2)
            self.assertTrue((canonical_root / "manifest.parquet").exists() or (canonical_root / "manifest.parquet.jsonl").exists())

            report = validate_canonical_root(canonical_root)
            self.assertTrue(report["ok"], json.dumps(report, indent=2))

            split = create_split(
                canonical_root,
                SplitRequest(name="unit", ratios={"train": 0.5, "holdout": 0.5}, seed=7, include_modalities={"gaze"}),
            )
            train = set(split["splits"]["train"])
            holdout = set(split["splits"]["holdout"])
            self.assertFalse(train & holdout)
            self.assertEqual(train | holdout, {"toy:ep1", "toy:ep2"})
            self.assertTrue((canonical_root / "splits" / "unit.json").exists())

    def test_alignment_validation_fails_on_gaze_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_root = tmp_path / "raw"
            canonical_root = tmp_path / "canonical"
            make_raw_episode(raw_root, "toy", "ep1")
            rectify_dataset(raw_root, canonical_root, default_config())

            episode_json = canonical_root / "episodes" / "toy" / "ep1" / "episode.json"
            doc = json.loads(episode_json.read_text())
            gaze_path = episode_json.parent / doc["files"]["gaze"]
            rows = read_table(gaze_path)
            rows[2]["x_norm"] = rows[2]["x_norm"] + 0.25
            write_table(rows, gaze_path)

            report = validate_canonical_root(canonical_root)
            self.assertFalse(report["ok"])
            errors = json.dumps(report)
            self.assertIn("gaze.x_norm differs", errors)

    def test_alignment_validation_fails_on_annotation_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_root = tmp_path / "raw"
            canonical_root = tmp_path / "canonical"
            make_raw_episode(raw_root, "toy", "ep1")
            rectify_dataset(raw_root, canonical_root, default_config())

            episode_json = canonical_root / "episodes" / "toy" / "ep1" / "episode.json"
            doc = json.loads(episode_json.read_text())
            annotations_path = episode_json.parent / doc["files"]["annotations"]
            rows = read_table(annotations_path)
            rows[1]["text"] = "mutated"
            write_table(rows, annotations_path)

            report = validate_canonical_root(canonical_root)
            self.assertFalse(report["ok"])
            self.assertIn("annotations.text changed", json.dumps(report))

    def test_alignment_validation_fails_on_timeline_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_root = tmp_path / "raw"
            canonical_root = tmp_path / "canonical"
            make_raw_episode(raw_root, "toy", "ep1")
            rectify_dataset(raw_root, canonical_root, default_config())

            episode_json = canonical_root / "episodes" / "toy" / "ep1" / "episode.json"
            doc = json.loads(episode_json.read_text())
            timeline_path = episode_json.parent / doc["files"]["timeline"]
            rows = read_table(timeline_path)
            rows.pop()
            write_table(rows, timeline_path)

            report = validate_canonical_root(canonical_root)
            self.assertFalse(report["ok"])
            self.assertIn("timeline sample count", json.dumps(report))


def make_raw_episode(raw_root: Path, dataset: str, episode_id: str) -> Path:
    root = raw_root / dataset / episode_id
    root.mkdir(parents=True, exist_ok=True)
    video = root / "video.mp4"
    video.write_bytes(b"synthetic video placeholder")
    write_csv(
        [
            {"time_s": 0.0, "x_norm": 0.1, "y_norm": 0.2, "x_px": 10, "y_px": 20},
            {"time_s": 0.1, "x_norm": 0.2, "y_norm": 0.3, "x_px": 20, "y_px": 30},
            {"time_s": 0.2, "x_norm": 0.3, "y_norm": 0.4, "x_px": 30, "y_px": 40},
            {"time_s": 0.3, "x_norm": 0.4, "y_norm": 0.5, "x_px": 40, "y_px": 50},
            {"time_s": 0.4, "x_norm": 0.5, "y_norm": 0.6, "x_px": 50, "y_px": 60},
        ],
        root / "gaze.csv",
    )
    write_csv(
        [
            {"start_s": 0.0, "end_s": 0.2, "label": "a", "text": "step a"},
            {"start_s": 0.2, "end_s": 0.4, "label": "b", "text": "step b"},
        ],
        root / "annotations.csv",
    )
    write_csv(
        [
            {"time_s": 0.0, "depth_m": 1.0},
            {"time_s": 0.2, "depth_m": 1.2},
            {"time_s": 0.4, "depth_m": 1.4},
        ],
        root / "depth.csv",
    )
    (root / "episode.json").write_text(
        json.dumps(
            {
                "dataset": dataset,
                "episode_id": episode_id,
                "duration_s": 0.4,
                "files": {
                    "video": "video.mp4",
                    "gaze": "gaze.csv",
                    "annotations": "annotations.csv",
                    "depth": "depth.csv",
                },
            },
            indent=2,
        )
    )
    return root


if __name__ == "__main__":
    unittest.main()
