"""Unit tests for the MolmoAct2 video-point emitter (synthetic, no remote/video)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gaze.curate import PadTransform
from src.gaze.molmoact import (
    build_molmoact_row,
    px_to_padded_px,
    sample_segment_frames,
)
from src.gaze.training import chop_into_segments


class TestPadGeometry(unittest.TestCase):
    def test_square_source_no_pad(self):
        pad = PadTransform(1408, 1408, 378)
        # center maps to center
        x, y = px_to_padded_px(pad, 704, 704)
        self.assertAlmostEqual(x, 189.0, places=1)
        self.assertAlmostEqual(y, 189.0, places=1)

    def test_wide_source_letterboxed(self):
        # 896x504 (16:9) -> fit width to 378, pad top/bottom
        pad = PadTransform(896, 504, 378)
        # a point at source (448,252)=center -> padded center (189,189)
        x, y = px_to_padded_px(pad, 448, 252)
        self.assertAlmostEqual(x, 189.0, places=1)
        self.assertAlmostEqual(y, 189.0, places=1)
        # top-left source -> x=0 but y offset by the pad band
        x0, y0 = px_to_padded_px(pad, 0, 0)
        self.assertAlmostEqual(x0, 0.0, places=1)
        self.assertGreater(y0, 0.0)  # letterbox band on top


class TestSampleSegmentFrames(unittest.TestCase):
    def _grid(self, n, fps=2.0):
        gt = [i / fps for i in range(n)]
        gpx = [100.0 + i * 10 for i in range(n)]
        gpy = [100.0 + i * 5 for i in range(n)]
        return gt, gpx, gpy, [True] * n, [True] * n

    def test_short_segment_pads(self):
        gt, gpx, gpy, gin, gv = self._grid(21)  # 0..10s @2fps
        pad = PadTransform(1408, 1408, 378)
        frames, num_real = sample_segment_frames(
            gt, gpx, gpy, gin, gv, seg_start_s=2.0, seg_end_s=5.0,
            fps=2.0, max_frames=8, pad=pad,
        )
        self.assertEqual(num_real, 6)          # 2.0,2.5,3.0,3.5,4.0,4.5
        self.assertEqual(len(frames), 8)       # padded to max_frames
        self.assertTrue(all(frames[i].valid for i in range(6)))
        self.assertFalse(any(frames[i].valid for i in range(6, 8)))

    def test_long_segment_caps(self):
        gt, gpx, gpy, gin, gv = self._grid(41)  # 0..20s @2fps
        pad = PadTransform(1408, 1408, 378)
        frames, num_real = sample_segment_frames(
            gt, gpx, gpy, gin, gv, seg_start_s=0.0, seg_end_s=20.0,
            fps=2.0, max_frames=8, pad=pad,
        )
        self.assertEqual(num_real, 8)          # capped
        self.assertEqual(len(frames), 8)

    def test_invalid_samples_masked(self):
        gt, gpx, gpy, gin, gv = self._grid(11)
        gv[4] = False  # invalid sample at index 4 (t=2.0)
        pad = PadTransform(1408, 1408, 378)
        frames, num_real = sample_segment_frames(
            gt, gpx, gpy, gin, gv, seg_start_s=1.0, seg_end_s=4.0,
            fps=2.0, max_frames=8, pad=pad,
        )
        # the t=2.0 frame should be present but invalid (point None)
        invalid = [f for f in frames[:num_real] if not f.valid]
        self.assertTrue(any(abs(f.t_s - 1.0) < 1e-6 for f in invalid))

    def test_empty_segment(self):
        gt, gpx, gpy, gin, gv = self._grid(5)
        pad = PadTransform(1408, 1408, 378)
        frames, num_real = sample_segment_frames(
            gt, gpx, gpy, gin, gv, seg_start_s=50.0, seg_end_s=55.0,
            fps=2.0, max_frames=8, pad=pad,
        )
        self.assertEqual(num_real, 0)
        self.assertEqual(frames, [])


class TestBuildRow(unittest.TestCase):
    def test_row_shape(self):
        gt = [i / 2.0 for i in range(13)]
        gpx = [200.0 + i * 30 for i in range(13)]
        gpy = [200.0 + i * 20 for i in range(13)]
        pad = PadTransform(1408, 1408, 378)
        frames, num_real = sample_segment_frames(
            gt, gpx, gpy, [True] * 13, [True] * 13,
            seg_start_s=0.0, seg_end_s=3.0, fps=2.0, max_frames=8, pad=pad,
        )
        row = build_molmoact_row(
            dataset="ego-exo4d", episode_id="cmu_bike01_2", seg_index=0,
            video_rel="videos/ego-exo4d/cmu_bike01_2__seg0.mp4",
            seg_start_s=0.0, seg_end_s=3.0, frames=frames, num_real=num_real,
            fps=2.0, max_frames=8, side=378,
            annotation_text="installs a wheel", prompt="Point to where the camera wearer is looking.",
        )
        self.assertEqual(row["style"], "video_point")
        self.assertEqual(row["num_frames"], 8)
        self.assertEqual(len(row["points"]), 8)
        self.assertEqual(len(row["timestamps"]), 8)
        self.assertEqual(len(row["frame_mask"]), 8)
        self.assertEqual(sum(row["frame_mask"]), num_real)
        # real frames carry one pixel point in [0,378]; pad frames are []
        for k in range(num_real):
            self.assertEqual(len(row["points"][k]), 1)
            p = row["points"][k][0]
            self.assertTrue(0 <= p["x"] <= 378 and 0 <= p["y"] <= 378)
        for k in range(num_real, 8):
            self.assertEqual(row["points"][k], [])
        # message_list chat shape
        self.assertEqual([m["role"] for m in row["message_list"]], ["user", "assistant"])
        self.assertEqual([c["type"] for c in row["message_list"][0]["content"]], ["text", "video"])
        self.assertEqual(row["metadata"]["clip_start_time"], 0.0)
        self.assertEqual(row["metadata"]["clip_end_time"], 3.0)

    def test_timestamps_on_half_second_grid(self):
        gt = [i / 2.0 for i in range(9)]
        gpx = [100.0] * 9
        gpy = [100.0] * 9
        pad = PadTransform(1408, 1408, 378)
        frames, num_real = sample_segment_frames(
            gt, gpx, gpy, [True] * 9, [True] * 9,
            seg_start_s=1.0, seg_end_s=4.0, fps=2.0, max_frames=8, pad=pad,
        )
        row = build_molmoact_row(
            dataset="d", episode_id="e", seg_index=0, video_rel="v.mp4",
            seg_start_s=1.0, seg_end_s=4.0, frames=frames, num_real=num_real,
            fps=2.0, max_frames=8, side=378, annotation_text=None, prompt="p",
        )
        # every timestamp is a multiple of 0.5 (the MolmoAct2 grid)
        for t in row["timestamps"]:
            self.assertAlmostEqual((t / 0.5) - round(t / 0.5), 0.0, places=6)


class TestChopIntegration(unittest.TestCase):
    def test_nymeria_like_offset_annotations(self):
        # annotations start ~98s in (the v3 bug); chop must place segments there.
        spans = [
            {"start_s": 98.0, "end_s": 128.0, "point_s": None},
            {"start_s": 128.0, "end_s": 160.0, "point_s": None},
        ]
        segs = chop_into_segments(spans, max_clip_s=20, merge_gap_s=2.0, min_clip_s=1.0, duration_s=1001)
        self.assertTrue(segs)
        self.assertGreaterEqual(segs[0]["start_s"], 98.0)
        self.assertTrue(all(s["end_s"] - s["start_s"] <= 20 + 1e-6 for s in segs))


if __name__ == "__main__":
    unittest.main()
