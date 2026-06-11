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
from src.gaze.training import (
    chop_into_segments,
    chop_by_channels,
    coalesce_short_segments,
    _numbered_text,
)


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
    """gaze_hz == video_fps: one gaze point per real video frame (1:1, no cap/pad)."""

    def _grid(self, n, fps=2.0):
        gt = [i / fps for i in range(n)]
        gpx = [100.0 + i * 10 for i in range(n)]
        gpy = [100.0 + i * 5 for i in range(n)]
        return gt, gpx, gpy, [True] * n, [True] * n

    def test_one_point_per_video_frame(self):
        # 3s segment @2fps encoded to 6 frames -> exactly 6 points, all real.
        gt, gpx, gpy, gin, gv = self._grid(21)  # 0..10s @2fps
        pad = PadTransform(1408, 1408, 378)
        frames, num_real = sample_segment_frames(
            gt, gpx, gpy, gin, gv, seg_start_s=2.0, seg_end_s=5.0,
            fps=2.0, n_frames=6, pad=pad,
        )
        self.assertEqual(len(frames), 6)       # exactly n_frames, no pad
        self.assertEqual(num_real, 6)
        self.assertTrue(all(f.valid for f in frames))
        # frame j is at clip-relative j/fps
        self.assertAlmostEqual(frames[0].t_s, 0.0)
        self.assertAlmostEqual(frames[1].t_s, 0.5)
        self.assertAlmostEqual(frames[-1].t_s, 2.5)

    def test_no_cap_long_segment(self):
        # 20s segment @2fps -> 40 frames; NO cap, all 40 emitted (gaze_hz==fps).
        gt, gpx, gpy, gin, gv = self._grid(41)  # 0..20s @2fps
        pad = PadTransform(1408, 1408, 378)
        frames, num_real = sample_segment_frames(
            gt, gpx, gpy, gin, gv, seg_start_s=0.0, seg_end_s=20.0,
            fps=2.0, n_frames=40, pad=pad,
        )
        self.assertEqual(len(frames), 40)
        self.assertEqual(num_real, 40)

    def test_frame_alignment_uses_segment_start(self):
        # gaze grid value at video-clock time = seg_start + j/fps. px = 100 + 10*grid_idx.
        gt, gpx, gpy, gin, gv = self._grid(41)
        pad = PadTransform(1408, 1408, 378)
        frames, _ = sample_segment_frames(
            gt, gpx, gpy, gin, gv, seg_start_s=2.0, seg_end_s=5.0,
            fps=2.0, n_frames=6, pad=pad,
        )
        # seg_start=2.0 @2fps -> grid index base=4; frame0 reads grid[4] (px=140).
        # px maps through pad (square source, side 378): x_px = 140/1408*378.
        self.assertAlmostEqual(frames[0].x_px, round(140.0 / 1408 * 378, 1), places=1)
        self.assertAlmostEqual(frames[1].x_px, round(150.0 / 1408 * 378, 1), places=1)

    def test_invalid_samples_masked_keep_slot(self):
        gt, gpx, gpy, gin, gv = self._grid(11)
        gv[2] = False  # invalid grid sample at index 2 (t=1.0)
        pad = PadTransform(1408, 1408, 378)
        frames, num_real = sample_segment_frames(
            gt, gpx, gpy, gin, gv, seg_start_s=1.0, seg_end_s=4.0,
            fps=2.0, n_frames=6, pad=pad,
        )
        # frame0 (seg_start=1.0 -> grid idx 2) is invalid but keeps its slot.
        self.assertEqual(len(frames), 6)
        self.assertFalse(frames[0].valid)
        self.assertIsNone(frames[0].x_px)
        self.assertEqual(num_real, 5)

    def test_frames_past_grid_end_are_masked(self):
        # request more frames than the grid covers -> trailing slots masked, kept.
        gt, gpx, gpy, gin, gv = self._grid(5)  # grid only to t=2.0
        pad = PadTransform(1408, 1408, 378)
        frames, num_real = sample_segment_frames(
            gt, gpx, gpy, gin, gv, seg_start_s=0.0, seg_end_s=5.0,
            fps=2.0, n_frames=10, pad=pad,
        )
        self.assertEqual(len(frames), 10)
        self.assertEqual(num_real, 5)          # grid had 5 samples
        self.assertFalse(any(f.valid for f in frames[5:]))

    def test_empty_grid(self):
        pad = PadTransform(1408, 1408, 378)
        frames, num_real = sample_segment_frames(
            [], [], [], [], [], seg_start_s=0.0, seg_end_s=5.0,
            fps=2.0, n_frames=10, pad=pad,
        )
        self.assertEqual(num_real, 0)
        self.assertEqual(frames, [])

    def test_zero_frames(self):
        gt, gpx, gpy, gin, gv = self._grid(5)
        pad = PadTransform(1408, 1408, 378)
        frames, num_real = sample_segment_frames(
            gt, gpx, gpy, gin, gv, seg_start_s=0.0, seg_end_s=5.0,
            fps=2.0, n_frames=0, pad=pad,
        )
        self.assertEqual(frames, [])
        self.assertEqual(num_real, 0)


class TestBuildRow(unittest.TestCase):
    def test_row_shape_one_point_per_frame(self):
        gt = [i / 2.0 for i in range(13)]
        gpx = [200.0 + i * 30 for i in range(13)]
        gpy = [200.0 + i * 20 for i in range(13)]
        pad = PadTransform(1408, 1408, 378)
        # 3s @2fps -> 6 frames, all valid (gaze_hz == video_fps, no cap/pad).
        frames, num_real = sample_segment_frames(
            gt, gpx, gpy, [True] * 13, [True] * 13,
            seg_start_s=0.0, seg_end_s=3.0, fps=2.0, n_frames=6, pad=pad,
        )
        row = build_molmoact_row(
            dataset="ego-exo4d", episode_id="cmu_bike01_2", seg_index=0,
            video_rel="videos/ego-exo4d/cmu_bike01_2__seg0.mp4",
            seg_start_s=0.0, seg_end_s=3.0, frames=frames, num_real=num_real,
            fps=2.0, side=378,
            annotation_text="installs a wheel", prompt="Point to where the camera wearer is looking.",
        )
        self.assertEqual(row["style"], "video_point")
        # num_frames == num points == len(frames); no fixed padding
        self.assertEqual(row["num_frames"], 6)
        self.assertEqual(len(row["points"]), 6)
        self.assertEqual(len(row["timestamps"]), 6)
        self.assertEqual(len(row["frame_mask"]), 6)
        self.assertEqual(row["num_frames_real"], num_real)
        self.assertEqual(sum(row["frame_mask"]), num_real)
        # every frame carries one pixel point in [0,378] (all valid here)
        for k in range(6):
            self.assertEqual(len(row["points"][k]), 1)
            p = row["points"][k][0]
            self.assertTrue(0 <= p["x"] <= 378 and 0 <= p["y"] <= 378)
        # message_list chat shape
        self.assertEqual([m["role"] for m in row["message_list"]], ["user", "assistant"])
        self.assertEqual([c["type"] for c in row["message_list"][0]["content"]], ["text", "video"])
        self.assertEqual(row["metadata"]["clip_start_time"], 0.0)
        self.assertEqual(row["metadata"]["clip_end_time"], 3.0)

    def test_invalid_frame_has_empty_point_and_mask0(self):
        gt = [i / 2.0 for i in range(9)]
        gpx = [100.0] * 9
        gpy = [100.0] * 9
        gv = [True] * 9
        gv[1] = False  # grid idx 1 invalid
        pad = PadTransform(1408, 1408, 378)
        frames, num_real = sample_segment_frames(
            gt, gpx, gpy, [True] * 9, gv,
            seg_start_s=0.0, seg_end_s=3.0, fps=2.0, n_frames=6, pad=pad,
        )
        row = build_molmoact_row(
            dataset="d", episode_id="e", seg_index=0, video_rel="v.mp4",
            seg_start_s=0.0, seg_end_s=3.0, frames=frames, num_real=num_real,
            fps=2.0, side=378, annotation_text=None, prompt="p",
        )
        self.assertEqual(len(row["points"]), 6)
        self.assertEqual(row["points"][1], [])         # invalid frame: empty
        self.assertEqual(row["frame_mask"][1], 0)
        self.assertEqual(row["frame_mask"][0], 1)
        self.assertEqual(sum(row["frame_mask"]), num_real)

    def test_timestamps_match_frame_index_over_fps(self):
        gt = [i / 2.0 for i in range(9)]
        gpx = [100.0] * 9
        gpy = [100.0] * 9
        pad = PadTransform(1408, 1408, 378)
        frames, num_real = sample_segment_frames(
            gt, gpx, gpy, [True] * 9, [True] * 9,
            seg_start_s=1.0, seg_end_s=4.0, fps=2.0, n_frames=6, pad=pad,
        )
        row = build_molmoact_row(
            dataset="d", episode_id="e", seg_index=0, video_rel="v.mp4",
            seg_start_s=1.0, seg_end_s=4.0, frames=frames, num_real=num_real,
            fps=2.0, side=378, annotation_text=None, prompt="p",
        )
        # timestamp of frame j == j/fps (clip-relative)
        for j, t in enumerate(row["timestamps"]):
            self.assertAlmostEqual(t, j / 2.0, places=6)


class TestHierarchicalChop(unittest.TestCase):
    def _channels(self):
        # coarsest -> finest, like holoassist (narration / coarse / fine)
        return [
            {"name": "narration", "kind": "interval", "mean_dur": 100.0,
             "spans": [{"start_s": 0.0, "end_s": 100.0, "text": "whole take"}]},
            {"name": "coarse", "kind": "interval", "mean_dur": 30.0,
             "spans": [{"start_s": 0.0, "end_s": 15.0, "text": "grabs gopro"},
                       {"start_s": 15.0, "end_s": 100.0, "text": "opens gopro"}]},
            {"name": "fine", "kind": "interval", "mean_dur": 3.0,
             "spans": [{"start_s": 16.0, "end_s": 19.0, "text": "rotate"},
                       {"start_s": 19.0, "end_s": 24.0, "text": "pull door"},
                       {"start_s": 24.0, "end_s": 30.0, "text": "grab battery"}]},
        ]

    def test_coarse_fits_used_else_descend(self):
        segs = chop_by_channels(self._channels(), max_clip_s=20, drop_shorter_than_s=1.0, duration_s=100)
        by_ch = [(s["channel"], round(s["start_s"], 1), round(s["end_s"], 1)) for s in segs]
        # the 0-15 coarse span fits (<=20) -> used as a coarse clip with its text
        self.assertIn(("coarse", 0.0, 15.0), by_ch)
        # the 15-100 coarse span is too long -> descend to fine within it
        fine = [s for s in segs if s["channel"] == "fine"]
        self.assertTrue(fine)
        self.assertTrue(all(s["end_s"] - s["start_s"] <= 20 + 1e-6 for s in segs))

    def test_clip_carries_driving_channel_text(self):
        segs = chop_by_channels(self._channels(), max_clip_s=20, drop_shorter_than_s=1.0, duration_s=100)
        coarse0 = next(s for s in segs if s["channel"] == "coarse" and s["start_s"] == 0.0)
        self.assertEqual(coarse0["text"], "grabs gopro")
        fine0 = next(s for s in segs if s["channel"] == "fine")
        self.assertIn(fine0["text"], {"rotate", "pull door", "grab battery"})

    def test_finest_too_long_hard_cuts(self):
        chans = [
            {"name": "only", "kind": "interval", "mean_dur": 50.0,
             "spans": [{"start_s": 0.0, "end_s": 50.0, "text": "long"}]},
        ]
        segs = chop_by_channels(chans, max_clip_s=20, drop_shorter_than_s=1.0, duration_s=50)
        self.assertTrue(all(s["end_s"] - s["start_s"] <= 20 + 1e-6 for s in segs))
        self.assertTrue(all(s["channel"] == "only" and s["text"] == "long" for s in segs))
        self.assertGreaterEqual(len(segs), 3)  # 50s / 20 -> 3 pieces

    def test_empty_channels(self):
        self.assertEqual(chop_by_channels([], max_clip_s=20, duration_s=10), [])


class TestNumberedText(unittest.TestCase):
    def test_single_unchanged(self):
        self.assertEqual(_numbered_text(["wash a plate"]), "wash a plate")

    def test_multiple_numbered(self):
        self.assertEqual(_numbered_text(["A", "B", "C"]), "1) A 2) B 3) C")

    def test_blanks_skipped(self):
        self.assertEqual(_numbered_text(["A", "", None, "B"]), "1) A 2) B")

    def test_consecutive_dupes_collapsed(self):
        self.assertEqual(_numbered_text(["walk", "walk", "talk"]), "1) walk 2) talk")

    def test_all_blank(self):
        self.assertEqual(_numbered_text(["", None]), "")


class TestCoalesceShortSegments(unittest.TestCase):
    def _segs(self, spans):
        # spans: list of (start, end, text)
        return [{"start_s": a, "end_s": b, "channel": "c", "text": t} for a, b, t in spans]

    def test_disabled_when_zero(self):
        segs = self._segs([(0, 1, "A"), (1, 2, "B")])
        out = coalesce_short_segments(segs, min_duration_s=0.0, max_clip_s=20)
        self.assertEqual(out, segs)

    def test_merges_short_into_next(self):
        # two 1s clips, min 2s -> merge into one 2s clip with numbered text
        segs = self._segs([(0, 1, "A"), (1, 2, "B")])
        out = coalesce_short_segments(segs, min_duration_s=2.0, max_clip_s=20)
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out[0]["start_s"], 0.0)
        self.assertAlmostEqual(out[0]["end_s"], 2.0)
        self.assertEqual(out[0]["text"], "1) A 2) B")
        self.assertEqual(out[0]["coalesced"], 2)

    def test_keeps_long_enough_alone(self):
        segs = self._segs([(0, 3, "A"), (3, 6, "B")])
        out = coalesce_short_segments(segs, min_duration_s=2.0, max_clip_s=20)
        self.assertEqual(len(out), 2)
        self.assertNotIn("coalesced", out[0])

    def test_absorbs_multiple_until_min_then_drops_trailing(self):
        # four 1s clips, min 3s -> first absorbs 2 more (0..3); trailing 1s clip D
        # cannot reach 3s and is DROPPED (item 3: never ship sub-min clips).
        segs = self._segs([(0, 1, "A"), (1, 2, "B"), (2, 3, "C"), (3, 4, "D")])
        out = coalesce_short_segments(segs, min_duration_s=3.0, max_clip_s=20)
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out[0]["end_s"] - out[0]["start_s"], 3.0)
        self.assertEqual(out[0]["text"], "1) A 2) B 3) C")

    def test_drop_unmergeable_single_short_clip(self):
        # one isolated 1s clip, min 3s, nothing to merge -> DROPPED by default
        segs = self._segs([(0, 1, "A")])
        self.assertEqual(coalesce_short_segments(segs, min_duration_s=3.0, max_clip_s=20), [])
        # keep when drop_unmergeable=False
        kept = coalesce_short_segments(segs, min_duration_s=3.0, max_clip_s=20, drop_unmergeable=False)
        self.assertEqual(len(kept), 1)

    def test_respects_max_clip_ceiling_drops_unfillable(self):
        # min 5s but max_clip 2s -> can never reach 5s -> all dropped by default
        segs = self._segs([(0, 1, "A"), (1, 2, "B"), (2, 3, "C")])
        out = coalesce_short_segments(segs, min_duration_s=5.0, max_clip_s=2.0)
        self.assertEqual(out, [])
        kept = coalesce_short_segments(segs, min_duration_s=5.0, max_clip_s=2.0, drop_unmergeable=False)
        for s in kept:
            self.assertLessEqual(s["end_s"] - s["start_s"], 2.0 + 1e-6)

    def test_gap_between_clips_preserved_in_span(self):
        # clips with a gap: merging extends end to absorbed clip's end (covers the gap)
        segs = self._segs([(0, 1, "A"), (5, 6, "B")])
        out = coalesce_short_segments(segs, min_duration_s=2.0, max_clip_s=20)
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out[0]["end_s"], 6.0)


class TestChannelRolesAndPoints(unittest.TestCase):
    """Point-channel widening + clip_role (driver/context/disabled) + auxiliary attach."""

    def _data(self, annos, recipe_annos):
        from src.gaze.overlay import EpisodeData
        return EpisodeData(
            slug="syn", episode_id="ep",
            video={"fps": 30.0, "width": 1408, "height": 1408, "duration_s": 100.0, "path": "x.mp4"},
            gaze_space="already_2d", gaze_rows=[], annotations=annos,
            epoch_sync={"annotations": {"transform": "as_is"}},
            projection_method="already_2d", projection={},
            recipe={"root": "x", "annotations": recipe_annos}, tok={},
        )

    def test_point_channel_widened_to_interval(self):
        from src.gaze.training import channels_by_granularity
        data = self._data(
            [{"name": "atomic", "kind": "point", "segments": [
                {"start_s": None, "end_s": None, "point_s": 10.0, "text": "grabs wheel"}]}],
            [{"name": "atomic", "clip_role": "driver"}],
        )
        chans = channels_by_granularity(data, point_window_s=1.0)
        self.assertEqual(len(chans), 1)
        sp = chans[0]["spans"][0]
        self.assertAlmostEqual(sp["start_s"], 9.0)   # point -/+ window
        self.assertAlmostEqual(sp["end_s"], 11.0)

    def test_disabled_channel_excluded_everywhere(self):
        from src.gaze.training import channels_by_granularity, auxiliary_channel_spans
        data = self._data(
            [{"name": "atomic", "kind": "point", "segments": [
                {"start_s": None, "end_s": None, "point_s": 10.0, "text": "grabs wheel"}]},
             {"name": "expert", "kind": "interval", "segments": [
                {"start_s": 0.0, "end_s": 50.0, "point_s": None, "text": "expert says..."}]}],
            [{"name": "atomic", "clip_role": "driver"}, {"name": "expert", "clip_role": "disabled"}],
        )
        chans = [c["name"] for c in channels_by_granularity(data)]
        self.assertEqual(chans, ["atomic"])                 # expert not a driver
        aux = {s["channel"] for s in auxiliary_channel_spans(data)}
        self.assertEqual(aux, {"atomic"})                    # expert not auxiliary either

    def test_context_channel_is_auxiliary_not_driver(self):
        from src.gaze.training import channels_by_granularity, auxiliary_channel_spans
        data = self._data(
            [{"name": "atomic_action", "kind": "interval", "segments": [
                {"start_s": 5.0, "end_s": 8.0, "point_s": None, "text": "stirs pot"}]},
             {"name": "activity_summary", "kind": "interval", "segments": [
                {"start_s": 0.0, "end_s": 30.0, "point_s": None, "text": "cooking"}]}],
            [{"name": "atomic_action", "clip_role": "driver"},
             {"name": "activity_summary", "clip_role": "context"}],
        )
        drivers = {c["name"] for c in channels_by_granularity(data)}
        self.assertEqual(drivers, {"atomic_action"})         # context doesn't drive
        aux = {s["channel"] for s in auxiliary_channel_spans(data)}
        self.assertEqual(aux, {"atomic_action", "activity_summary"})  # but IS auxiliary

    def test_annotations_covering_clip_default_excluded(self):
        from src.gaze.training import annotations_covering_clip
        aux = [
            {"channel": "atomic_action", "text": "stirs pot", "start_s": 5.0, "end_s": 8.0},
            {"channel": "activity_summary", "text": "cooking dinner", "start_s": 0.0, "end_s": 30.0},
        ]
        # clip [5,8] driven by atomic_action "stirs pot" -> default excluded, activity aux kept
        cov = annotations_covering_clip(aux, 5.0, 8.0, driver_channel="atomic_action", driver_text="stirs pot")
        chans = [c["channel"] for c in cov]
        self.assertIn("activity_summary", chans)
        self.assertNotIn("atomic_action", chans)             # the driver is the default
        # times are clip-relative
        a = next(c for c in cov if c["channel"] == "activity_summary")
        self.assertAlmostEqual(a["start_s"], 0.0)            # max(0, 0-5)=0
        self.assertAlmostEqual(a["end_s"], 3.0)              # min(3, 30-5)=3


class TestChopIntegration(unittest.TestCase):
    def test_nymeria_like_offset_annotations(self):
        # annotations start ~98s in (the v3 bug); chop must place segments there.
        spans = [
            {"start_s": 98.0, "end_s": 128.0, "point_s": None},
            {"start_s": 128.0, "end_s": 160.0, "point_s": None},
        ]
        segs = chop_into_segments(spans, max_clip_s=20, merge_gap_s=2.0, drop_shorter_than_s=1.0, duration_s=1001)
        self.assertTrue(segs)
        self.assertGreaterEqual(segs[0]["start_s"], 98.0)
        self.assertTrue(all(s["end_s"] - s["start_s"] <= 20 + 1e-6 for s in segs))


if __name__ == "__main__":
    unittest.main()
