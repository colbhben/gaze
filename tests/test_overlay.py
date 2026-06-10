"""Unit tests for the overlay renderer's clock/lookup logic + projection helpers.

All tests here are synthetic and offline: no remote NFS, no video, no heavy
deps beyond what overlay.py imports lazily. They cover the load-bearing
reconciliation (frame -> video time -> nearest gaze sample) and the
annotation-active-at-t lookup, plus the pure-math projection helpers.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.gaze import overlay as ov
from src.gaze.overlay import (
    EpisodeData,
    ProjectionContext,
    active_segments,
    clip_start_of,
    frame_time,
    nearest_gaze_index,
    project_one,
    reconcile_to_video_clock,
)


class TestFrameTime(unittest.TestCase):
    def test_frame_time(self):
        self.assertAlmostEqual(frame_time(0, 30.0), 0.0)
        self.assertAlmostEqual(frame_time(30, 30.0), 1.0)
        self.assertAlmostEqual(frame_time(12, 24.0), 0.5)
        self.assertAlmostEqual(frame_time(15, 15.0), 1.0)

    def test_frame_time_no_fps(self):
        with self.assertRaises(ValueError):
            frame_time(1, 0)


class TestReconcileClock(unittest.TestCase):
    def test_as_is(self):
        self.assertAlmostEqual(reconcile_to_video_clock(5.0, "as_is"), 5.0)

    def test_subtract_first_gaze(self):
        # nymeria: gaze device clock t0 = 13649.395; a sample at 13747.85 -> 98.45s
        self.assertAlmostEqual(
            reconcile_to_video_clock(13747.845, "subtract_first_gaze", gaze_t0=13649.395),
            98.45, places=3,
        )

    def test_subtract_clip_start(self):
        # egtea: annotation ms-based seconds 1002.316 minus clip start 1002.316 -> 0
        self.assertAlmostEqual(
            reconcile_to_video_clock(1002.316, "subtract_clip_start", clip_start_s=1002.316),
            0.0, places=6,
        )
        self.assertAlmostEqual(
            reconcile_to_video_clock(1004.005, "subtract_clip_start", clip_start_s=1002.316),
            1.689, places=6,
        )

    def test_none_passthrough(self):
        self.assertIsNone(reconcile_to_video_clock(None, "as_is"))
        self.assertIsNone(reconcile_to_video_clock(None, "subtract_first_gaze", gaze_t0=1.0))

    def test_missing_anchor_raises(self):
        with self.assertRaises(ValueError):
            reconcile_to_video_clock(5.0, "subtract_first_gaze")
        with self.assertRaises(ValueError):
            reconcile_to_video_clock(5.0, "subtract_clip_start")

    def test_unknown_transform(self):
        with self.assertRaises(ValueError):
            reconcile_to_video_clock(5.0, "warp_drive")

    def test_clip_start_of_egtea(self):
        self.assertAlmostEqual(
            clip_start_of("egtea", "OP01-R01-PastaSalad-1002316-1004005-F024051-F024101"),
            1002.316,
        )
        self.assertIsNone(clip_start_of("nymeria", "anything"))


class TestNearestGaze(unittest.TestCase):
    def test_picks_nearest(self):
        # 10Hz grid 0,0.1,...; frame at t=0.52 -> index 5 (0.5)
        times = [i * 0.1 for i in range(10)]
        self.assertEqual(nearest_gaze_index(times, 0.52), 5)
        self.assertEqual(nearest_gaze_index(times, 0.58), 6)
        self.assertEqual(nearest_gaze_index(times, 0.0), 0)
        self.assertEqual(nearest_gaze_index(times, 5.0), 9)  # beyond -> last

    def test_empty(self):
        self.assertIsNone(nearest_gaze_index([], 1.0))
        self.assertIsNone(nearest_gaze_index([None, None], 1.0))

    def test_max_dt_gap(self):
        # a gaze gap: nearest sample 3s away with max_dt=0.5 -> None (no stale dot)
        times = [0.0, 0.1, 0.2]
        self.assertIsNone(nearest_gaze_index(times, 3.0, max_dt=0.5))
        self.assertEqual(nearest_gaze_index(times, 0.25, max_dt=0.5), 2)

    def test_skips_none_entries(self):
        times = [0.0, None, 0.2, None, 0.4]
        self.assertEqual(nearest_gaze_index(times, 0.21), 2)
        self.assertEqual(nearest_gaze_index(times, 0.39), 4)

    def test_valid_mask_excludes_invalid(self):
        # invalid sample at the nearest index must not be selected (egome/egoexolearn
        # placeholder/dropout rows): pick the nearest VALID neighbor instead.
        times = [0.0, 1.0, 2.0, 3.0, 4.0]
        mask = [True, True, False, True, True]
        self.assertIn(nearest_gaze_index(times, 2.0, max_dt=5.0, valid_mask=mask), (1, 3))
        # unmasked still picks the (invalid) nearest
        self.assertEqual(nearest_gaze_index(times, 2.0, max_dt=5.0), 2)

    def test_valid_mask_invalid_run_blanks(self):
        # an invalid run longer than max_dt around t -> None (blank dot, not stale).
        times = [0.0, 1.0, 2.0, 3.0, 4.0]
        mask = [True, False, False, False, True]
        self.assertIsNone(nearest_gaze_index(times, 2.0, max_dt=0.5, valid_mask=mask))

    def test_valid_mask_all_invalid(self):
        times = [0.0, 1.0, 2.0]
        self.assertIsNone(nearest_gaze_index(times, 1.0, valid_mask=[False, False, False]))


class TestActiveSegments(unittest.TestCase):
    def test_interval_covering(self):
        segs = [
            {"start_s": 0.0, "end_s": 2.0, "text": "a"},
            {"start_s": 1.0, "end_s": 3.0, "text": "b"},   # overlaps a
            {"start_s": 5.0, "end_s": 6.0, "text": "c"},
        ]
        act = active_segments(segs, 1.5, kind="interval")
        self.assertEqual({s["text"] for s in act}, {"a", "b"})
        self.assertEqual([s["text"] for s in active_segments(segs, 5.5, kind="interval")], ["c"])
        self.assertEqual(active_segments(segs, 4.0, kind="interval"), [])  # gap -> empty

    def test_interval_boundaries_inclusive(self):
        segs = [{"start_s": 1.0, "end_s": 2.0, "text": "x"}]
        self.assertEqual(len(active_segments(segs, 1.0, kind="interval")), 1)
        self.assertEqual(len(active_segments(segs, 2.0, kind="interval")), 1)
        self.assertEqual(len(active_segments(segs, 2.01, kind="interval")), 0)

    def test_point_nearest_within_tol(self):
        segs = [{"point_s": 2.6, "text": "p1"}, {"point_s": 9.96, "text": "p2"}]
        # ego-exo4d atomic: a point shows for +-point_tol around it
        self.assertEqual([s["text"] for s in active_segments(segs, 2.7, kind="point")], ["p1"])
        self.assertEqual(active_segments(segs, 5.0, kind="point"), [])  # too far
        self.assertEqual([s["text"] for s in active_segments(segs, 10.0, kind="point")], ["p2"])

    def test_missing_bounds_skipped(self):
        segs = [{"start_s": None, "end_s": 5.0, "text": "bad"},
                {"start_s": 1.0, "end_s": None, "text": "open"}]
        # None start skipped; open interval shows for point_tol after start
        self.assertEqual([s["text"] for s in active_segments(segs, 1.2, kind="interval")], ["open"])
        self.assertEqual(active_segments(segs, 4.0, kind="interval"), [])


class TestProjectOnePure(unittest.TestCase):
    """The non-aria/non-psi projection paths are pure math -> test offline."""

    def test_already_2d(self):
        ctx = ProjectionContext(method="already_2d", width=1408, height=1408)
        self.assertEqual(project_one({"x": 700.0, "y": 650.0}, ctx), (700.0, 650.0))
        self.assertIsNone(project_one({"x": None, "y": 1.0}, ctx))

    def test_normalize_normalized_space(self):
        ctx = ProjectionContext(method="normalize_by_dims", width=320, height=320,
                                coordinate_space="normalized_2d")
        self.assertEqual(project_one({"x": 0.5, "y": 0.5}, ctx), (160.0, 160.0))

    def test_normalize_pixel_rescale(self):
        # source 1280x960 -> mp4 640x480 (scale 0.5)
        ctx = ProjectionContext(method="normalize_by_dims", width=640, height=480,
                                coordinate_space="pixel_2d", frame_dims=(1280.0, 960.0))
        x, y = project_one({"x": 640.0, "y": 480.0}, ctx)
        self.assertAlmostEqual(x, 320.0)
        self.assertAlmostEqual(y, 240.0)

    def test_missing_returns_none(self):
        ctx = ProjectionContext(method="normalize_by_dims", width=320, height=320,
                                coordinate_space="normalized_2d")
        self.assertIsNone(project_one({"x": None, "y": None}, ctx))


class TestReconcileWholeTable(unittest.TestCase):
    """End-to-end clock reconciliation over a synthetic EpisodeData."""

    def _make(self, gaze_transform, anno_transform, slug="syn", episode_id="ep"):
        # gaze device clock starts at 100.0; samples at 100.0, 100.1, 100.2
        rows = [{"t_s": 100.0 + i * 0.1, "x": 0.5, "y": 0.5} for i in range(3)]
        annos = [{"name": "ch", "kind": "interval",
                  "segments": [{"start_s": 100.05, "end_s": 100.25, "text": "act"}]}]
        return EpisodeData(
            slug=slug, episode_id=episode_id,
            video={"fps": 30.0, "width": 320, "height": 320, "duration_s": 1.0, "path": "x.mp4"},
            gaze_space="normalized_2d", gaze_rows=rows, annotations=annos,
            epoch_sync={"gaze": {"transform": gaze_transform},
                        "annotations": {"transform": anno_transform}},
            projection_method="normalize_by_dims", projection={}, recipe={"root": "x"}, tok={},
        )

    def test_subtract_first_gaze_aligns_gaze_and_anno(self):
        data = self._make("subtract_first_gaze", "subtract_first_gaze")
        gt = ov.reconciled_gaze_times(data)
        self.assertAlmostEqual(gt[0], 0.0)
        self.assertAlmostEqual(gt[1], 0.1)
        annos = ov.reconciled_annotations(data)
        seg = annos[0]["segments"][0]
        self.assertAlmostEqual(seg["start_s"], 0.05)
        self.assertAlmostEqual(seg["end_s"], 0.25)
        # frame 3 of a 30fps clip -> t=0.1 -> nearest gaze idx 1; anno active
        t = frame_time(3, 30.0)
        self.assertEqual(nearest_gaze_index(gt, t), 1)
        self.assertEqual(len(active_segments(annos[0]["segments"], t, kind="interval")), 1)

    def test_as_is_keeps_device_clock(self):
        data = self._make("as_is", "as_is")
        gt = ov.reconciled_gaze_times(data)
        self.assertAlmostEqual(gt[0], 100.0)


if __name__ == "__main__":
    unittest.main()
