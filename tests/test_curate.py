"""Tests for the recipe-driven extraction harness (curate.py + curate_readers.py).

Covers:
  * load_recipe works for all 7 datasets (+ defaults merge).
  * to_seconds unit conversions (s/ms/us/ns/filetime_100ns/frame_index/hms/timecode).
  * compose_text: strip_prefix and compose.
  * annotation_stats: interval-union coverage.
  * eval_predicate / valid_when declarative predicates.
  * curate_readers imports and the gaze + annotation reader dispatch resolves
    for every recipe; begaze variant matching; psi/normalize/already projection
    math on synthetic fixtures (no remote needed).
  * A remote-dependent smoke test for extract_episode that SKIPS when ssh to the
    data host is unavailable.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import unittest
from pathlib import Path

import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.gaze import curate
from src.gaze import curate_readers as cr
from src.gaze.curate import (
    annotation_stats,
    compose_text,
    load_recipe,
    to_seconds,
    GazeTable,
    VideoMeta,
)

SLUGS = ["ego-exo4d", "egoexolearn", "egome", "egtea", "holoassist", "nymeria", "hd-epic"]


class TestRecipeLoading(unittest.TestCase):
    def test_load_recipe_all_seven(self):
        for slug in SLUGS:
            rec = load_recipe(slug)
            self.assertEqual(rec["dataset"], slug, slug)
            self.assertIn("root", rec)
            self.assertIn("gaze", rec)
            self.assertIn("annotations", rec)
            self.assertGreaterEqual(len(rec["annotations"]), 1, slug)
            # defaults merged in
            self.assertEqual(rec["recipe_version"], 1, slug)
            self.assertIn("emit_policy", rec)

    def test_gaze_reader_dispatch_resolves(self):
        for slug in SLUGS:
            rec = load_recipe(slug)
            reader = rec["gaze"]["gaze_format"]["reader"]
            self.assertIn(reader, cr.GAZE_READERS, f"{slug}:{reader}")

    def test_annotation_reader_dispatch_resolves(self):
        valid = {"csv", "json_by_key", "pandas_pickle", "json", "tsv"}
        for slug in SLUGS:
            rec = load_recipe(slug)
            for ch in rec["annotations"]:
                self.assertIn(ch["reader"], valid, f"{slug}:{ch['name']}")

    def test_projection_methods_known(self):
        known = {"already_2d", "normalize_by_dims", "projectaria_cpf", "psi_pinhole_ray", "none"}
        for slug in SLUGS:
            rec = load_recipe(slug)
            m = rec["gaze"]["gaze_format"]["projection"]["method"]
            self.assertIn(m, known, f"{slug}:{m}")


class TestToSeconds(unittest.TestCase):
    def test_basic_units(self):
        self.assertAlmostEqual(to_seconds(2.5, "s"), 2.5)
        self.assertAlmostEqual(to_seconds(2500, "ms"), 2.5)
        self.assertAlmostEqual(to_seconds(2_500_000, "us"), 2.5)
        self.assertAlmostEqual(to_seconds(2_500_000_000, "ns"), 2.5)

    def test_filetime_100ns(self):
        # 1 second = 1e7 ticks of 100ns
        self.assertAlmostEqual(to_seconds(1e7, "filetime_100ns"), 1.0)
        self.assertAlmostEqual(to_seconds(637932482246358268, "filetime_100ns"),
                               637932482246358268 / 1e7)

    def test_frame_index(self):
        self.assertAlmostEqual(to_seconds(30, "frame_index", fps=30), 1.0)
        self.assertAlmostEqual(to_seconds(12, "frame_index", fps=24), 0.5)
        with self.assertRaises(ValueError):
            to_seconds(30, "frame_index")  # missing fps

    def test_hms_decimal(self):
        self.assertAlmostEqual(to_seconds("00:00:07.340", "hms_decimal"), 7.340)
        self.assertAlmostEqual(to_seconds("01:02:03.5", "hms_decimal"), 3723.5)

    def test_timecode_hmsf(self):
        # HH:MM:SS:FF, FF frames within second
        self.assertAlmostEqual(to_seconds("00:00:01:12", "timecode_hmsf", fps=24), 1.5)
        self.assertAlmostEqual(to_seconds("00:01:00:00", "timecode_hmsf", fps=24), 60.0)

    def test_none_and_empty(self):
        self.assertIsNone(to_seconds(None, "s"))
        self.assertIsNone(to_seconds("", "s"))

    def test_unknown_units(self):
        with self.assertRaises(ValueError):
            to_seconds(1, "fortnights")


class TestComposeText(unittest.TestCase):
    def test_primary(self):
        self.assertEqual(compose_text({"text": "hello"}, {"primary": "text"}), "hello")

    def test_strip_prefix(self):
        # egome 'C:'/'T:'/'F:' prefixes
        self.assertEqual(
            compose_text({"Coarse-level": "C:Press both buttons."},
                         {"primary": "Coarse-level", "strip_prefix_regex": "^[A-Z]:"}),
            "Press both buttons.",
        )

    def test_compose(self):
        # holoassist fine Verb+Adjective+Noun
        row = {"attributes": {"Verb": "cut", "Adjective": "red", "Noun": "tomato"}}
        spec = {"compose": ["attributes.Verb", "attributes.Adjective", "attributes.Noun"],
                "compose_sep": " "}
        self.assertEqual(compose_text(row, spec), "cut red tomato")

    def test_compose_drops_none(self):
        row = {"a": "x", "b": None, "c": "z"}
        spec = {"compose": ["a", "b", "c"], "compose_sep": " "}
        self.assertEqual(compose_text(row, spec), "x z")

    def test_dotted_primary(self):
        row = {"attributes": {"Long form description": "a long thing"}}
        self.assertEqual(
            compose_text(row, {"primary": "attributes.Long form description"}),
            "a long thing",
        )


class TestAnnotationStats(unittest.TestCase):
    def test_interval_union_overlapping(self):
        segs = [
            {"start_s": 0.0, "end_s": 10.0},
            {"start_s": 5.0, "end_s": 12.0},   # overlaps -> union [0,12]
            {"start_s": 20.0, "end_s": 25.0},  # disjoint -> +5
        ]
        ch = annotation_stats("x", "interval", segs, video_duration_s=50.0)
        self.assertEqual(ch.segment_count, 3)
        self.assertAlmostEqual(ch.coverage_s, 17.0)  # 12 + 5
        self.assertAlmostEqual(ch.coverage_fraction, 17.0 / 50.0)
        self.assertAlmostEqual(ch.first_start_s, 0.0)
        self.assertAlmostEqual(ch.last_end_s, 25.0)

    def test_interval_union_adjacent(self):
        segs = [{"start_s": 0.0, "end_s": 5.0}, {"start_s": 5.0, "end_s": 9.0}]
        ch = annotation_stats("x", "interval", segs, video_duration_s=10.0)
        self.assertAlmostEqual(ch.coverage_s, 9.0)

    def test_point_channel_no_coverage(self):
        segs = [{"point_s": 1.0}, {"point_s": 3.0}, {"point_s": 7.5}]
        ch = annotation_stats("p", "point", segs, video_duration_s=10.0)
        self.assertEqual(ch.segment_count, 3)
        self.assertIsNone(ch.coverage_s)
        self.assertAlmostEqual(ch.first_start_s, 1.0)
        self.assertAlmostEqual(ch.last_end_s, 7.5)


class TestPredicates(unittest.TestCase):
    def test_eq_str_and_bool(self):
        self.assertTrue(cr.eval_predicate({"view": "ego"}, "view=='ego'"))
        self.assertFalse(cr.eval_predicate({"view": "exo"}, "view=='ego'"))
        self.assertTrue(cr.eval_predicate({"rejected": True}, "rejected==true"))
        self.assertFalse(cr.eval_predicate({"error": False}, "error==true"))

    def test_dotted_field(self):
        self.assertTrue(cr.eval_predicate({"label": "Narration"}, "label=='Narration'"))

    def test_empty_is_true(self):
        self.assertTrue(cr.eval_predicate({}, None))
        self.assertTrue(cr.eval_predicate({}, ""))

    def test_valid_when(self):
        self.assertTrue(cr._eval_valid(1, "==1"))
        self.assertFalse(cr._eval_valid(0, "==1"))
        self.assertTrue(cr._eval_valid(5.0, ">=0"))
        self.assertFalse(cr._eval_valid(-1.0, ">=0"))


class TestDatasetFilters(unittest.TestCase):
    def test_cull_from_recipe(self):
        # egoexolearn is culled; ego-exo4d is not (real recipes)
        self.assertTrue(cr.is_culled("egoexolearn"))
        self.assertFalse(cr.is_culled("ego-exo4d"))

    def test_exclude_globs(self):
        self.assertTrue(cr.episode_excluded("ego-exo4d", "cmu_soccer_01_2"))
        self.assertTrue(cr.episode_excluded("ego-exo4d", "nus_cpr_5"))
        self.assertTrue(cr.episode_excluded("ego-exo4d", "upenn_0718_Violin_2_3"))  # mid-pattern glob
        self.assertFalse(cr.episode_excluded("ego-exo4d", "cmu_bike01_2"))
        self.assertFalse(cr.episode_excluded("ego-exo4d", "fair_cooking_05_2"))

    def test_filter_episode_ids(self):
        self.assertEqual(cr.filter_episode_ids("egoexolearn", ["a", "b"]), [])  # culled -> all dropped
        self.assertEqual(
            cr.filter_episode_ids("ego-exo4d", ["cmu_bike01_2", "nus_soccer_3", "fair_cooking_05_2"]),
            ["cmu_bike01_2", "fair_cooking_05_2"],
        )

    def test_max_gaze_gap(self):
        exceeded, mx = cr.max_gaze_gap_exceeded([0.0, 0.1, 0.5, 0.6], [True, True, True, True], 0.25)
        self.assertTrue(exceeded)
        self.assertAlmostEqual(mx, 0.4, places=6)
        exceeded, mx = cr.max_gaze_gap_exceeded([0.0, 0.1, 0.2], [True, True, True], 0.25)
        self.assertFalse(exceeded)
        # invalid samples don't count toward the valid-to-valid gap
        exceeded, mx = cr.max_gaze_gap_exceeded([0.0, 0.1, 0.5], [True, False, True], 0.25)
        self.assertTrue(exceeded)  # valid 0.0 -> 0.5 gap = 0.5


class TestPullerLocalRoot(unittest.TestCase):
    """local_root (e.g. /nfs on the data host) reads sources IN PLACE -- no copy,
    and the in-place source is never owned (so cleanup can't delete the mount)."""

    def setUp(self):
        import tempfile
        self.tmp = Path(tempfile.mkdtemp(prefix="gaze_puller_test_"))
        self.mount = self.tmp / "mount"
        (self.mount / "sub").mkdir(parents=True)
        self.src = self.mount / "sub" / "video.mp4"
        self.src.write_bytes(b"x" * 1024)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_pull_reads_in_place_no_copy(self):
        from src.gaze.curate import Puller
        p = Puller(local_root=self.mount, workdir=self.tmp / "work")
        got = p.pull("sub/video.mp4")
        # returns the mount path directly, not a copy under workdir
        self.assertEqual(got.resolve(), self.src.resolve())
        self.assertFalse((self.tmp / "work" / "sub" / "video.mp4").exists())

    def test_in_place_source_not_owned(self):
        from src.gaze.curate import Puller
        p = Puller(local_root=self.mount, workdir=self.tmp / "work")
        got = p.pull("sub/video.mp4")
        self.assertFalse(p.owns(got))  # never delete the read-only mount source

    def test_workdir_outputs_are_owned(self):
        from src.gaze.curate import Puller
        p = Puller(local_root=self.mount, workdir=self.tmp / "work")
        (p.workdir / "seg0.mp4").write_bytes(b"y")
        self.assertTrue(p.owns(p.workdir / "seg0.mp4"))

    def test_missing_local_source_raises(self):
        from src.gaze.curate import Puller
        p = Puller(local_root=self.mount, workdir=self.tmp / "work")
        with self.assertRaises(FileNotFoundError):
            p.pull("sub/nope.mp4")

    def test_exists_and_glob_local(self):
        from src.gaze.curate import Puller
        p = Puller(local_root=self.mount, workdir=self.tmp / "work")
        self.assertTrue(p.exists("sub/video.mp4"))
        self.assertFalse(p.exists("sub/nope.mp4"))
        self.assertEqual(p.glob("sub/*.mp4"), ["sub/video.mp4"])


class TestPathHelpers(unittest.TestCase):
    def test_path_get_list_index(self):
        rec = {"Step timestamp": [1.5, 9.0]}
        self.assertEqual(cr._path_get(rec, "Step timestamp.0"), 1.5)
        self.assertEqual(cr._path_get(rec, "Step timestamp.1"), 9.0)

    def test_col_get_paren_suffix(self):
        row = {"Clip Prefix (Unique)": "OP01-x", "Action Label": "Cut"}
        self.assertEqual(cr._col_get(row, "Clip Prefix"), "OP01-x")
        self.assertEqual(cr._col_get(row, "Action Label"), "Cut")

    def test_egtea_frame_range(self):
        self.assertEqual(
            cr._egtea_frame_range("OP01-R01-PastaSalad-1002316-1004005-F024051-F024101"),
            (24051, 24101),
        )

    def test_episode_tokens(self):
        tok = cr._episode_tokens("hd-epic", "P01-20240202-110250", {})
        self.assertEqual(tok["participant"], "P01")
        tok2 = cr._episode_tokens("egtea", "OP01-R01-PastaSalad-1-2-F3-F4", {})
        self.assertEqual(tok2["session"], "OP01-R01-PastaSalad")


class TestGazeReaders(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(__file__).resolve().parent / "_curate_fixtures"
        self.tmp.mkdir(exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_csv_already_2d(self):
        p = self.tmp / "g.csv"
        p.write_text("frame_num,tracking_timestamp_us,x,y\n0,1000000,700,650\n1,1100000,710,655\n")
        gf = {
            "reader": "csv", "coordinate_space": "pixel_2d", "frame_dims": [1408, 1408],
            "columns": {"x": "x", "y": "y", "frame": "frame_num"},
            "time": {"source": "tracking_timestamp_us", "units": "us"},
            "projection": {"method": "already_2d"},
        }
        gt = cr.read_gaze_csv(p, gf, fps=30)
        self.assertEqual(gt.sample_count, 2)
        self.assertEqual(gt.coordinate_space, "pixel_2d")
        self.assertAlmostEqual(gt.rows[0]["x"], 700.0)
        self.assertAlmostEqual(gt.rows[0]["t_s"], 1.0)

    def test_npy_with_validity(self):
        import numpy as np
        arr = np.array([[0.5, 0.5, 0], [0.25, 0.75, 1], [0.6, 0.4, 1]], dtype="float32")
        p = self.tmp / "g.npy"
        np.save(p, arr)
        gf = {
            "reader": "npy", "coordinate_space": "normalized_2d",
            "columns": {"x": 0, "y": 1},
            "time": {"source": "frame_index", "units": "frame_index", "fps": 30},
            "validity": {"column": 2, "valid_when": "==1"},
            "projection": {"method": "normalize_by_dims"},
        }
        gt = cr.read_gaze_npy(p, gf, fps=30)
        self.assertEqual(gt.sample_count, 3)
        self.assertFalse(gt.rows[0]["valid"])  # validity 0 placeholder
        self.assertTrue(gt.rows[1]["valid"])
        self.assertAlmostEqual(gt.valid_fraction, 2 / 3, places=5)
        self.assertAlmostEqual(gt.rows[1]["t_s"], 1 / 30)

    def test_whitespace_txt(self):
        p = self.tmp / "eyes.txt"
        p.write_text(
            "0.0\t637\t0.1\t2.7\t0.01\t0.65\t0.22\t-0.72\t1\n"
            "0.0333\t637\t0.1\t2.7\t0.01\t0.69\t0.24\t-0.68\t0\n"
        )
        gf = {
            "reader": "whitespace_txt", "coordinate_space": "head_ray_3d",
            "columns": {"px": 2, "py": 3, "pz": 4, "vx": 5, "vy": 6, "vz": 7},
            "time": {"source": 0, "units": "s"},
            "validity": {"column": 8, "valid_when": "==1"},
            "extra_channels": [1],
            "projection": {"method": "psi_pinhole_ray"},
        }
        gt = cr.read_gaze_whitespace_txt(p, gf, fps=30)
        self.assertEqual(gt.sample_count, 2)
        self.assertTrue(gt.rows[0]["valid"])
        self.assertFalse(gt.rows[1]["valid"])
        self.assertAlmostEqual(gt.rows[0]["px"], 0.1)

    def test_begaze_variant_match_and_slice(self):
        # 3.1 layout with an in-file header row; frame range slice.
        p = self.tmp / "sess.txt"
        p.write_text(
            "## [BeGaze]\n## Version:\tBeGaze 3.1.77\n## Sample Rate:\t30\n"
            "Time\tType\tTrial\tL POR X [px]\tL POR Y [px]\tFrame\tAux1\tL Event Info\n"
            "100\tSMP\t1\t471.2\t465.0\t1\t\tFixation\n"
            "133\tSMP\t1\t473.9\t464.5\t2\t\tFixation\n"
            "166\tSMP\t1\t476.1\t463.5\t3\t\tFixation\n"
        )
        gf = {
            "reader": "begaze_txt", "coordinate_space": "pixel_2d", "frame_dims": [1280, 960],
            "columns": {"x": 3, "y": 4, "frame": 5, "type": 6},
            "time": {"source": 5, "units": "frame_index", "fps": 24},
            "projection": {"method": "normalize_by_dims"},
            "variants": [
                {"version": "3.1", "match": "## Version: BeGaze 3.1",
                 "columns": {"x": 3, "y": 4, "frame": 5, "type": 6},
                 "time": {"source": 5, "units": "frame_index", "fps": 24}},
                {"version": "3.4", "match": "## Version: BeGaze 3.4",
                 "columns": {"x": 5, "y": 6, "frame": -2, "type": -1},
                 "time": {"source": -2, "units": "timecode_hmsf", "fps": 24}},
            ],
        }
        gt = cr.read_gaze_begaze_txt(p, gf, fps=24, frame_range=(2, 3))
        self.assertEqual(gt.sample_count, 2)  # frames 2 and 3 only
        self.assertAlmostEqual(gt.rows[0]["x"], 473.9)


class TestProjection(unittest.TestCase):
    def test_normalize_by_dims_pixel_rescale(self):
        # source pixel space 1280x960, mp4 640x480 -> scale 0.5
        gaze = GazeTable(
            coordinate_space="pixel_2d", columns=["x", "y"],
            rows=[{"x": 640.0, "y": 480.0, "t_s": 0.0, "valid": True}],
            projection={"method": "normalize_by_dims", "_frame_dims": [1280, 960]},
        )
        video = VideoMeta(path="x", width=640, height=480, fps=24, duration_s=2.0)
        out = cr.project_gaze(gaze, video, puller=None, root="x", tok={}, n_samples=1)
        s = out["samples"][0]
        self.assertAlmostEqual(s["x_px"], 320.0)
        self.assertAlmostEqual(s["y_px"], 240.0)
        self.assertTrue(s["in_frame"])

    def test_normalize_by_dims_normalized_to_pixels(self):
        gaze = GazeTable(
            coordinate_space="normalized_2d", columns=["x", "y"],
            rows=[{"x": 0.5, "y": 0.5, "t_s": 0.0, "valid": True}],
            projection={"method": "normalize_by_dims"},
        )
        video = VideoMeta(path="x", width=320, height=320, fps=25, duration_s=2.0)
        out = cr.project_gaze(gaze, video, puller=None, root="x", tok={}, n_samples=1)
        s = out["samples"][0]
        self.assertAlmostEqual(s["x_px"], 160.0)
        self.assertAlmostEqual(s["y_px"], 160.0)

    def test_already_2d_passthrough(self):
        gaze = GazeTable(
            coordinate_space="pixel_2d", columns=["x", "y"],
            rows=[{"x": 700.0, "y": 650.0, "t_s": 1.0, "valid": True}],
            projection={"method": "already_2d"},
        )
        video = VideoMeta(path="x", width=1408, height=1408, fps=30, duration_s=10.0)
        out = cr.project_gaze(gaze, video, puller=None, root="x", tok={}, n_samples=1)
        s = out["samples"][0]
        self.assertAlmostEqual(s["x_px"], 700.0)
        self.assertTrue(s["in_frame"])

    def test_psi_intrinsics_parse(self):
        # the exact 24-float Intrinsics.txt seen on R0027-12-GoPro
        tmp = Path(__file__).resolve().parent / "_intr.txt"
        tmp.write_text(
            "681.3881225585938\t0\t445.44842529296875\t0\t682.3858642578125\t237.43148803710938"
            "\t0\t0\t1\t0\t0\t0\t0\t0\t0\t0\t0\t681.8869934082031\t681.3881225585938"
            "\t682.3858642578125\t445.44842529296875\t237.43148803710938\t1\t896\t504"
        )
        try:
            intr = cr._parse_psi_intrinsics(tmp)
            self.assertAlmostEqual(intr["flx"], 681.388, places=2)
            self.assertAlmostEqual(intr["fly"], 682.386, places=2)
            self.assertAlmostEqual(intr["ppx"], 445.448, places=2)
            self.assertAlmostEqual(intr["ppy"], 237.431, places=2)
            self.assertEqual(intr["w"], 896)
            self.assertEqual(intr["h"], 504)
        finally:
            tmp.unlink(missing_ok=True)


def _ssh_available(host: str = "sumedhso-L40S") -> bool:
    if os.environ.get("GAZE_SKIP_REMOTE"):
        return False
    if shutil.which("ssh") is None:
        return False
    try:
        r = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=8", "-o", "BatchMode=yes", host, "true"],
            capture_output=True, timeout=15,
        )
        return r.returncode == 0
    except Exception:
        return False


@unittest.skipUnless(_ssh_available(), "remote data host unreachable (set GAZE_SKIP_REMOTE to force skip)")
class TestExtractEpisodeRemote(unittest.TestCase):
    """Smoke test that extract_episode resolves+parses a small real episode."""

    def test_egtea_clip(self):
        from src.gaze.curate import Puller
        puller = Puller(workdir="/tmp/gaze_curate_test_work")
        bundle = cr.extract_episode(
            "egtea", "OP01-R01-PastaSalad-1002316-1004005-F024051-F024101", puller,
        )
        self.assertIsNotNone(bundle.video)
        self.assertIsNotNone(bundle.gaze)
        self.assertGreater(bundle.gaze.sample_count, 0)
        self.assertTrue(any(a.segment_count > 0 for a in bundle.annotations))
        self.assertTrue(bundle.emitted, bundle.emit_reason)


if __name__ == "__main__":
    unittest.main()
