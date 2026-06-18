"""Unit tests for the Qwen3-VL training-manifest builder (gaze-8yc.4).

All synthetic and offline: no remote NFS, no video, no heavy deps. Covers the
load-bearing math -- gaze resample-to-fps + linear interpolation, the [0,1]->0-1000
bin mapping with clamping, the pad-transform normalization, sliding-window clip
construction (counts / anchor times / edge handling / temporalities), and the
active-interval annotation lookup.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.gaze import training as tr
from src.gaze.training import (
    PadTransform,
    ResampledGaze,
    active_annotation_at,
    build_episode_examples,
    clamp01,
    clip_frame_indices,
    resample_track_linear,
    to_bins,
    unified_annotation_spans,
)
from src.gaze.overlay import EpisodeData


# =========================================================================== #
# Linear resampling onto the fps grid.
# =========================================================================== #
class TestResampleLinear(unittest.TestCase):
    def test_grid_times_and_count(self):
        # 30 Hz source over 1s, resample to 5 Hz over a 1s clip -> grid 0,0.2,...,1.0
        times = [i / 30.0 for i in range(31)]
        xs = [float(i) for i in range(31)]
        ys = [0.0] * 31
        rg = resample_track_linear(times, xs, ys, [True] * 31,
                                   fps=5.0, duration_s=1.0, max_gap_s=1.0)
        self.assertEqual(len(rg.times_s), 6)
        self.assertAlmostEqual(rg.times_s[0], 0.0)
        self.assertAlmostEqual(rg.times_s[-1], 1.0)

    def test_linear_interpolation_value(self):
        # samples at t=0 (x=0) and t=1 (x=10); midpoint t=0.5 -> 5.0
        rg = resample_track_linear([0.0, 1.0], [0.0, 10.0], [0.0, 0.0], [True, True],
                                   fps=2.0, duration_s=1.0, max_gap_s=2.0)
        # grid 0.0,0.5,1.0
        self.assertAlmostEqual(rg.px[0], 0.0)
        self.assertAlmostEqual(rg.px[1], 5.0)   # linear interp
        self.assertAlmostEqual(rg.px[2], 10.0)
        self.assertTrue(all(rg.valid))

    def test_exact_sample_hit(self):
        rg = resample_track_linear([0.0, 0.2, 0.4], [1.0, 2.0, 3.0], [0.0, 0.0, 0.0],
                                   [True, True, True], fps=5.0, duration_s=0.4, max_gap_s=1.0)
        self.assertAlmostEqual(rg.px[0], 1.0)
        self.assertAlmostEqual(rg.px[1], 2.0)
        self.assertAlmostEqual(rg.px[2], 3.0)

    def test_no_extrapolation_outside_track(self):
        # track only covers [0.4, 0.6]; grid points before/after stay invalid
        rg = resample_track_linear([0.4, 0.6], [4.0, 6.0], [0.0, 0.0], [True, True],
                                   fps=5.0, duration_s=1.0, max_gap_s=1.0)
        # grid 0.0,0.2,0.4,0.6,0.8,1.0
        self.assertFalse(rg.valid[0])  # 0.0 before track
        self.assertFalse(rg.valid[1])  # 0.2 before track
        self.assertTrue(rg.valid[2])   # 0.4 == first
        self.assertTrue(rg.valid[3])   # 0.6 == last
        self.assertFalse(rg.valid[4])  # 0.8 after track
        self.assertIsNone(rg.px[0])

    def test_gap_wider_than_max_not_interpolated(self):
        # 2s gap with max_gap_s=1.0 -> grid points inside the gap stay invalid
        rg = resample_track_linear([0.0, 2.0], [0.0, 20.0], [0.0, 0.0], [True, True],
                                   fps=1.0, duration_s=2.0, max_gap_s=1.0)
        # grid 0,1,2
        self.assertTrue(rg.valid[0])    # exact first sample
        self.assertFalse(rg.valid[1])   # inside the 2s gap
        self.assertTrue(rg.valid[2])    # exact last sample
        self.assertIsNone(rg.px[1])

    def test_in_frame_is_or_of_bracketing(self):
        rg = resample_track_linear([0.0, 1.0], [0.0, 10.0], [0.0, 0.0], [False, True],
                                   fps=2.0, duration_s=1.0, max_gap_s=2.0)
        # midpoint brackets a False + True -> True (conservative)
        self.assertTrue(rg.in_frame[1])

    def test_empty_track(self):
        rg = resample_track_linear([], [], [], [], fps=5.0, duration_s=1.0, max_gap_s=1.0)
        self.assertEqual(len(rg.times_s), 6)
        self.assertTrue(all(v is None for v in rg.px))
        self.assertFalse(any(rg.valid))


# =========================================================================== #
# [0,1] -> 0-1000 mapping + clamping; pad-transform normalization.
# =========================================================================== #
class TestBinsAndClamp(unittest.TestCase):
    def test_clamp01(self):
        self.assertEqual(clamp01(-0.3), 0.0)
        self.assertEqual(clamp01(1.7), 1.0)
        self.assertEqual(clamp01(0.42), 0.42)

    def test_to_bins_basic(self):
        self.assertEqual(to_bins(0.0), 0)
        self.assertEqual(to_bins(1.0), 1000)
        self.assertEqual(to_bins(0.5), 500)
        self.assertEqual(to_bins(0.3721), 372)  # round

    def test_to_bins_clamps_out_of_range(self):
        self.assertEqual(to_bins(-0.5), 0)
        self.assertEqual(to_bins(1.5), 1000)

    def test_to_bins_in_range(self):
        for v in (0.0, 0.123, 0.5, 0.999, 1.0):
            b = to_bins(v)
            self.assertGreaterEqual(b, 0)
            self.assertLessEqual(b, 1000)


class TestPadTransform(unittest.TestCase):
    def test_square_source_no_letterbox(self):
        # 1408x1408 -> 392 square: scale 392/1408, no padding offsets
        pad = PadTransform(src_w=1408, src_h=1408, side=392)
        self.assertAlmostEqual(pad.offset_x, 0.0)
        self.assertAlmostEqual(pad.offset_y, 0.0)
        xn, yn = pad.px_to_norm(704.0, 704.0)  # centre
        self.assertAlmostEqual(xn, 0.5, places=6)
        self.assertAlmostEqual(yn, 0.5, places=6)

    def test_wide_source_letterboxed_top_bottom(self):
        # 1280x960 (4:3) -> 392: scale = 392/1280, content_h < 392, pad top/bottom
        pad = PadTransform(src_w=1280, src_h=960, side=392)
        self.assertAlmostEqual(pad.scale, 392 / 1280, places=9)
        self.assertAlmostEqual(pad.offset_x, 0.0)
        self.assertGreater(pad.offset_y, 0.0)  # vertical bars
        # source centre maps to padded centre x, but y shifted by the letterbox
        xn, yn = pad.px_to_norm(640.0, 480.0)
        self.assertAlmostEqual(xn, 0.5, places=6)
        self.assertAlmostEqual(yn, 0.5, places=6)
        # top-left corner of source content sits at (0, offset_y) in padded px
        xn0, yn0 = pad.px_to_norm(0.0, 0.0)
        self.assertAlmostEqual(xn0, 0.0, places=6)
        self.assertAlmostEqual(yn0, pad.offset_y / 392, places=6)

    def test_tall_source_letterboxed_left_right(self):
        pad = PadTransform(src_w=504, src_h=896, side=392)
        self.assertGreater(pad.offset_x, 0.0)
        self.assertAlmostEqual(pad.offset_y, 0.0)


# =========================================================================== #
# Sliding-window clip construction.
# =========================================================================== #
class TestClipFrameIndices(unittest.TestCase):
    def test_causal_anchor_is_last(self):
        idx = clip_frame_indices(15, 16, temporality="causal")
        self.assertEqual(idx, list(range(0, 16)))
        self.assertEqual(idx[-1], 15)  # anchor is last

    def test_future_anchor_is_first(self):
        idx = clip_frame_indices(10, 16, temporality="future")
        self.assertEqual(idx[0], 10)
        self.assertEqual(idx[-1], 25)

    def test_centered(self):
        idx = clip_frame_indices(10, 8, temporality="centered")
        self.assertEqual(idx, list(range(6, 14)))

    def test_unknown_raises(self):
        with self.assertRaises(ValueError):
            clip_frame_indices(0, 8, temporality="sideways")


class TestBuildExamples(unittest.TestCase):
    def _data(self, slug="syn", ep="ep"):
        return EpisodeData(
            slug=slug, episode_id=ep,
            video={"fps": 5.0, "width": 392, "height": 392, "duration_s": 10.0, "path": "x.mp4"},
            gaze_space="already_2d", gaze_rows=[], annotations=[],
            epoch_sync={}, projection_method="already_2d", projection={}, recipe={"root": "x"}, tok={},
        )

    def _resampled(self, n, fps=5.0, valid_from=0, valid_to=None):
        valid_to = n if valid_to is None else valid_to
        times = [i / fps for i in range(n)]
        px = [196.0 if (valid_from <= i < valid_to) else None for i in range(n)]
        py = [196.0 if (valid_from <= i < valid_to) else None for i in range(n)]
        inf = [valid_from <= i < valid_to for i in range(n)]
        valid = [valid_from <= i < valid_to for i in range(n)]
        return ResampledGaze(fps, times, px, py, inf, valid)

    def test_causal_drops_leading_partial_clips(self):
        # 50 grid frames, 16-frame causal clips, stride 8 -> first anchor must be >=15
        data = self._data()
        rg = self._resampled(50)
        pad = PadTransform(392, 392, 392)
        ex = build_episode_examples(
            data, rg, [], pad, video_rel="v.mp4", fps=5.0, num_frames=16, stride=8,
            temporality="causal", n_video_frames=50,
        )
        anchors = [e["anchor_frame_index"] for e in ex]
        self.assertEqual(anchors[0], 16)  # 0,8 dropped (clip would start <0); 16 is first valid
        self.assertTrue(all(a >= 15 for a in anchors))
        self.assertTrue(all(a <= 49 for a in anchors))
        # all frame index lists are monotonic and in range
        for e in ex:
            self.assertEqual(e["frame_indices"], sorted(e["frame_indices"]))
            self.assertGreaterEqual(e["frame_indices"][0], 0)
            self.assertLessEqual(e["frame_indices"][-1], 49)

    def test_anchor_time_matches_index_over_fps(self):
        data = self._data()
        rg = self._resampled(40)
        pad = PadTransform(392, 392, 392)
        ex = build_episode_examples(
            data, rg, [], pad, video_rel="v.mp4", fps=5.0, num_frames=16, stride=8,
            temporality="causal", n_video_frames=40,
        )
        for e in ex:
            self.assertAlmostEqual(e["anchor_time_s"], e["anchor_frame_index"] / 5.0, places=6)
            self.assertAlmostEqual(e["frame_times_s"][-1], e["anchor_time_s"], places=6)

    def test_future_drops_trailing_partial_clips(self):
        data = self._data()
        rg = self._resampled(40)
        pad = PadTransform(392, 392, 392)
        ex = build_episode_examples(
            data, rg, [], pad, video_rel="v.mp4", fps=5.0, num_frames=16, stride=8,
            temporality="future", n_video_frames=40,
        )
        # future: anchor+15 must be <= 39 -> last anchor <= 24
        self.assertTrue(all(e["anchor_frame_index"] <= 24 for e in ex))
        self.assertEqual(ex[0]["anchor_frame_index"], 0)

    def test_missing_anchor_gaze_dropped(self):
        data = self._data()
        # valid only for indices [16, 32); causal clips at anchors 16,24 valid, others dropped
        rg = self._resampled(40, valid_from=16, valid_to=32)
        pad = PadTransform(392, 392, 392)
        ex = build_episode_examples(
            data, rg, [], pad, video_rel="v.mp4", fps=5.0, num_frames=16, stride=8,
            temporality="causal", n_video_frames=40,
        )
        anchors = [e["anchor_frame_index"] for e in ex]
        self.assertEqual(anchors, [16, 24])  # 32 invalid-> dropped; 8 partial-clip dropped

    def test_target_valid_and_bins_in_range(self):
        data = self._data()
        rg = self._resampled(40)
        pad = PadTransform(392, 392, 392)
        ex = build_episode_examples(
            data, rg, [], pad, video_rel="v.mp4", fps=5.0, num_frames=16, stride=8,
            temporality="causal", n_video_frames=40,
        )
        for e in ex:
            self.assertTrue(0.0 <= e["target"][0] <= 1.0)
            self.assertTrue(0.0 <= e["target"][1] <= 1.0)
            self.assertTrue(0 <= e["target_1000"][0] <= 1000)
            self.assertTrue(0 <= e["target_1000"][1] <= 1000)
            self.assertTrue(e["target_valid"])
            # 196/392 = 0.5 -> 500 bin
            self.assertEqual(e["target_1000"], [500, 500])

    def test_out_of_frame_kept_with_valid_false(self):
        data = self._data()
        # project to a px outside the 392 frame (e.g. 500,500) but still "present"
        n = 40
        times = [i / 5.0 for i in range(n)]
        px = [500.0] * n  # outside [0,392]
        py = [500.0] * n
        inf = [False] * n   # out of frame
        valid = [True] * n  # but the sample exists
        rg = ResampledGaze(5.0, times, px, py, inf, valid)
        pad = PadTransform(392, 392, 392)
        ex = build_episode_examples(
            data, rg, [], pad, video_rel="v.mp4", fps=5.0, num_frames=16, stride=8,
            temporality="causal", n_video_frames=40,
        )
        self.assertTrue(len(ex) > 0)
        for e in ex:
            self.assertFalse(e["target_valid"])      # kept, but flagged
            self.assertFalse(e["target_in_frame"])
            self.assertTrue(0.0 <= e["target"][0] <= 1.0)  # still clamped to [0,1]


# =========================================================================== #
# Annotation unification + active_interval lookup.
# =========================================================================== #
class TestAnnotationUnification(unittest.TestCase):
    def _data_with_annos(self, annos):
        return EpisodeData(
            slug="syn", episode_id="ep",
            video={"fps": 5.0, "width": 392, "height": 392, "duration_s": 10.0, "path": "x.mp4"},
            gaze_space="already_2d", gaze_rows=[], annotations=annos,
            epoch_sync={"annotations": {"transform": "as_is"}},
            projection_method="already_2d", projection={}, recipe={"root": "x"}, tok={},
        )

    def test_unify_flattens_channels_with_provenance(self):
        data = self._data_with_annos([
            {"name": "coarse", "kind": "interval",
             "segments": [{"start_s": 0.0, "end_s": 5.0, "point_s": None, "text": "make salad"}]},
            {"name": "fine", "kind": "interval",
             "segments": [{"start_s": 1.0, "end_s": 2.0, "point_s": None, "text": "cut tomato"},
                          {"start_s": 1.5, "end_s": 3.0, "point_s": None, "text": ""}]},  # empty dropped
        ])
        spans = unified_annotation_spans(data)
        self.assertEqual(len(spans), 2)  # the empty-text span is dropped
        channels = {s["channel"] for s in spans}
        self.assertEqual(channels, {"coarse", "fine"})
        for s in spans:
            self.assertIn("kind", s)
            self.assertIn("text", s)

    def test_active_interval_returns_all_covering(self):
        spans = [
            {"start_s": 0.0, "end_s": 5.0, "point_s": None, "text": "make salad", "channel": "coarse", "kind": "interval"},
            {"start_s": 1.0, "end_s": 2.0, "point_s": None, "text": "cut tomato", "channel": "fine", "kind": "interval"},
        ]
        # at t=1.5 both cover
        active = active_annotation_at(spans, 1.5)
        self.assertEqual({s["text"] for s in active}, {"make salad", "cut tomato"})
        # at t=4.0 only the coarse span covers
        active = active_annotation_at(spans, 4.0)
        self.assertEqual([s["text"] for s in active], ["make salad"])
        # at t=8.0 nothing
        self.assertEqual(active_annotation_at(spans, 8.0), [])

    def test_active_point_nearest_within_tol(self):
        spans = [
            {"start_s": None, "end_s": None, "point_s": 2.6, "text": "p1", "channel": "atomic", "kind": "point"},
            {"start_s": None, "end_s": None, "point_s": 9.0, "text": "p2", "channel": "atomic", "kind": "point"},
        ]
        self.assertEqual([s["text"] for s in active_annotation_at(spans, 2.7)], ["p1"])
        self.assertEqual(active_annotation_at(spans, 5.0), [])  # too far from any point
        self.assertEqual([s["text"] for s in active_annotation_at(spans, 9.1)], ["p2"])

    def test_interval_preferred_first_in_ordering(self):
        spans = [
            {"start_s": None, "end_s": None, "point_s": 1.5, "text": "pt", "channel": "atomic", "kind": "point"},
            {"start_s": 0.0, "end_s": 5.0, "point_s": None, "text": "iv", "channel": "coarse", "kind": "interval"},
        ]
        active = active_annotation_at(spans, 1.5)
        self.assertEqual(active[0]["kind"], "interval")  # intervals sorted first


class TestEpisodeListNoSampleContamination(unittest.TestCase):
    """An explicit episodes-file must process ONLY its listed datasets -- no fallback to
    per-recipe sample episodes for unlisted slugs (the old 'sample-default contamination'
    that forced a post-hoc join). Lets ONE build produce ONE clean manifest natively."""

    def test_only_listed_datasets_scheduled(self):
        import tempfile
        from pathlib import Path
        scheduled = []

        def fake_build(slug, ep, extra, out_root, puller, **kw):
            scheduled.append((slug, ep))
            return [], {"dataset": slug, "episode": ep}

        orig = tr._build_molmo2_episode
        tr._build_molmo2_episode = fake_build
        try:
            out = Path(tempfile.mkdtemp())
            tr.build_training_manifest(
                out, output_format="molmo2",
                episode_lists={"holoassist": ["R0027-12-GoPro"]},
                fps=6, max_clip_s=16, drop_shorter_than_s=0, min_duration_s=2.5,
                workers=1, puller=tr.Puller(local_root="/tmp"),
            )
        finally:
            tr._build_molmo2_episode = orig
        self.assertEqual({s for s, _ in scheduled}, {"holoassist"})  # listed only, no strays


class TestShardUnionAssembly(unittest.TestCase):
    """Manifest is assembled from EVERY shard on disk (glob of _shards/*.json), not just
    this process's jobs list -- so multi-process / multi-slice runs sharing one out-root
    produce the COMPLETE manifest, not one slice's subset (the partial-overwrite bug)."""

    def test_assembly_unions_all_shards_not_just_jobs(self):
        import json
        import tempfile
        from pathlib import Path

        out = Path(tempfile.mkdtemp())
        sd = out / "_shards"
        sd.mkdir(parents=True)
        def _row(ds, ep):  # minimal molmo2 row the manifest writers accept
            return {"id": f"{ds}:{ep}#seg0", "dataset": ds, "episode_id": ep,
                    "seg_index": 0, "num_frames": 3, "metadata": {}}

        # Pre-seed 5 shards as if written by OTHER slice processes (NOT in this call's jobs).
        for i in range(5):
            (sd / f"holoassist__ep{i}.json").write_text(json.dumps({
                "examples": [_row("holoassist", f"ep{i}")],
                "report": {"dataset": "holoassist", "episode": f"ep{i}", "clips": 1},
            }), encoding="utf-8")
        # also a *.tmp partial that must be IGNORED by the *.json glob
        (sd / "holoassist__epX.123.tmp").write_text('{"examples":[{"id":"x"}]}', encoding="utf-8")

        # This build's jobs is a DIFFERENT single episode; its worker is a no-op stub.
        def fake_build(slug, ep, extra, out_root, puller, **kw):
            return [_row(slug, ep)], {"dataset": slug, "episode": ep, "clips": 1}

        orig = tr._build_molmo2_episode
        tr._build_molmo2_episode = fake_build
        try:
            rep = tr.build_training_manifest(
                out, output_format="molmo2",
                episode_lists={"holoassist": ["ep_new"]},   # 1 job; 5 pre-seeded shards on disk
                fps=6, max_clip_s=16, drop_shorter_than_s=0, min_duration_s=2.5,
                workers=1, puller=tr.Puller(local_root="/tmp"),
            )
        finally:
            tr._build_molmo2_episode = orig

        rows = (out / "manifest.jsonl").read_text().strip().splitlines()
        # 5 pre-seeded + 1 new = 6 (the .tmp partial is excluded)
        self.assertEqual(rep["total_examples"], 6)
        self.assertEqual(rep["shards_assembled"], 6)
        self.assertEqual(len(rows), 6)


if __name__ == "__main__":
    unittest.main()
