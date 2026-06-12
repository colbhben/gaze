"""Unit tests for clip-level, per-dataset-stratified split pointers."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gaze.splits_manifest import (
    build_split,
    iter_clip_pointers,
    join_manifests,
    make_stratified_splits,
)


def _ptrs(spec):
    # spec: {dataset: n_clips} -> pointer list
    out = []
    for ds, n in spec.items():
        for i in range(n):
            out.append({"id": f"{ds}:ep{i}#seg0", "dataset": ds, "video": f"videos/{ds}/{i}.mp4"})
    return out


class TestStratifiedSplit(unittest.TestCase):
    def test_ratio_applied_per_dataset(self):
        ptrs = _ptrs({"a": 100, "b": 100})
        s = make_stratified_splits(ptrs, {"train": 0.8, "val": 0.2}, seed=1)
        # 80/20 within EACH dataset
        for ds in ("a", "b"):
            tr = sum(1 for p in s["train"] if p["dataset"] == ds)
            va = sum(1 for p in s["val"] if p["dataset"] == ds)
            self.assertEqual(tr, 80)
            self.assertEqual(va, 20)

    def test_minority_dataset_in_val(self):
        # a huge dataset + a tiny 5-clip one: val must still see the minority
        ptrs = _ptrs({"big": 1000, "tiny": 5})
        s = make_stratified_splits(ptrs, {"train": 0.8, "val": 0.2}, seed=2)
        val_ds = {p["dataset"] for p in s["val"]}
        self.assertIn("tiny", val_ds)            # minority guaranteed in val
        self.assertEqual(sum(1 for p in s["val"] if p["dataset"] == "tiny"), 1)   # 20% of 5 = 1

    def test_partition_is_exact_and_disjoint(self):
        ptrs = _ptrs({"a": 37, "b": 63})
        s = make_stratified_splits(ptrs, {"train": 0.7, "val": 0.3}, seed=3)
        all_ids = [p["id"] for sp in s.values() for p in sp]
        self.assertEqual(len(all_ids), 100)              # every clip assigned
        self.assertEqual(len(set(all_ids)), 100)         # exactly once (disjoint)

    def test_deterministic(self):
        ptrs = _ptrs({"a": 50, "b": 30})
        s1 = make_stratified_splits(ptrs, {"train": 0.8, "val": 0.2}, seed=7)
        s2 = make_stratified_splits(ptrs, {"train": 0.8, "val": 0.2}, seed=7)
        self.assertEqual([p["id"] for p in s1["val"]], [p["id"] for p in s2["val"]])
        # different seed -> different val membership (very likely)
        s3 = make_stratified_splits(ptrs, {"train": 0.8, "val": 0.2}, seed=8)
        self.assertNotEqual(sorted(p["id"] for p in s1["val"]), sorted(p["id"] for p in s3["val"]))

    def test_three_way_and_normalization(self):
        ptrs = _ptrs({"a": 100})
        s = make_stratified_splits(ptrs, {"train": 70, "val": 15, "test": 15}, seed=1)  # un-normalized
        self.assertEqual(len(s["train"]), 70)
        self.assertEqual(len(s["val"]), 15)
        self.assertEqual(len(s["test"]), 15)

    def test_clip_level_not_take_level(self):
        # two takes of the same dataset, many clips each; splitting is per-CLIP so clips
        # from one take can land in both train and val.
        ptrs = [{"id": f"d:take{t}#seg{c}", "dataset": "d", "video": "v"} for t in range(2) for c in range(50)]
        s = make_stratified_splits(ptrs, {"train": 0.8, "val": 0.2}, seed=5)
        self.assertEqual(len(s["train"]), 80)
        self.assertEqual(len(s["val"]), 20)


class TestPointersAndJoin(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def _manifest(self, rows):
        p = self.tmp / "m.jsonl"
        p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        return p

    def test_iter_pointers_drops_heavy_fields_and_errors(self):
        m = self._manifest([
            {"id": "a:e#seg0", "dataset": "a", "video": "v0.mp4", "points": [[{"x": 1, "y": 2}]] * 50,
             "message_list": [{"role": "user"}]},
            {"error": "boom", "dataset": "a"},          # skipped
            {"dataset": "a"},                            # no id -> skipped
        ])
        ptrs = list(iter_clip_pointers(m))
        self.assertEqual(len(ptrs), 1)
        self.assertEqual(set(ptrs[0]), {"id", "dataset", "video"})   # only pointer fields
        self.assertNotIn("points", ptrs[0])

    def test_dataset_fallback_from_id(self):
        m = self._manifest([{"id": "holoassist:x#seg0", "video": "v"}])  # no dataset field
        ptrs = list(iter_clip_pointers(m))
        self.assertEqual(ptrs[0]["dataset"], "holoassist")

    def test_join_keeps_only_requested_datasets_and_dedupes(self):
        m1 = self._manifest([
            {"id": "ego:1#0", "dataset": "ego-exo4d", "video": "a"},
            {"id": "nym:stray#0", "dataset": "nymeria", "video": "b"},   # stray, dropped
        ])
        m2 = (self.tmp / "m2.jsonl")
        m2.write_text("".join(json.dumps(r) + "\n" for r in [
            {"id": "nym:1#0", "dataset": "nymeria", "video": "c"},
            {"id": "ego:1#0", "dataset": "ego-exo4d", "video": "dup"},   # dup id -> first wins, dropped here
        ]), encoding="utf-8")
        out = self.tmp / "joint.jsonl"
        rep = join_manifests([(m1, {"ego-exo4d"}), (m2, {"nymeria"})], out)
        ids = [json.loads(l)["id"] for l in out.read_text().splitlines()]
        self.assertEqual(sorted(ids), ["ego:1#0", "nym:1#0"])   # stray nymeria + dup dropped
        self.assertEqual(rep["by_dataset"], {"ego-exo4d": 1, "nymeria": 1})

    def test_build_split_end_to_end(self):
        m = self._manifest(
            [{"id": f"a:e{i}#0", "dataset": "a", "video": f"{i}.mp4"} for i in range(80)]
            + [{"id": f"b:e{i}#0", "dataset": "b", "video": f"{i}.mp4"} for i in range(20)]
        )
        idx = build_split(m, self.tmp / "splits", name="s1", ratios={"train": 0.8, "val": 0.2}, seed=0)
        base = self.tmp / "splits" / "s1"
        self.assertTrue((base / "train.jsonl").exists())
        self.assertTrue((base / "val.jsonl").exists())
        self.assertTrue((base / "split_index.json").exists())
        # val has both datasets; counts match
        self.assertEqual(idx["counts"]["val"]["by_dataset"], {"a": 16, "b": 4})
        self.assertEqual(idx["counts"]["train"]["by_dataset"], {"a": 64, "b": 16})
        self.assertEqual(idx["join_key"], "id")
        # pointer files contain only pointer fields
        first = json.loads((base / "val.jsonl").read_text().splitlines()[0])
        self.assertEqual(set(first), {"id", "dataset", "video"})


if __name__ == "__main__":
    unittest.main()
