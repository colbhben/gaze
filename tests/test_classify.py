"""Unit tests for the interesting-region classification helpers (stdlib, no model)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gaze.classify import (
    build_filter_map,
    classify_prompt,
    export_annotation_spans,
    parse_verdicts,
)


def _bundle():
    return {"annotations": [{"name": "atomic_action", "kind": "interval", "segments": [
        {"start_s": 10.0, "end_s": 15.0, "text": "C stands in the kitchen talking to peer"},
        {"start_s": 15.0, "end_s": 20.0, "text": "C picks up a pan and stirs the contents"},
        {"start_s": 20.0, "end_s": 25.0, "text": ""},  # empty -> dropped
    ]}]}


class TestExport(unittest.TestCase):
    def test_drops_empty_text(self):
        rec = export_annotation_spans(_bundle(), "take1")
        self.assertEqual(len(rec["spans"]), 2)
        self.assertEqual([s["i"] for s in rec["spans"]], [0, 1])
        self.assertEqual(rec["take_id"], "take1")


class TestPrompt(unittest.TestCase):
    def test_prompt_lists_spans_and_rubric(self):
        rec = export_annotation_spans(_bundle(), "take1")
        p = classify_prompt(rec)
        self.assertIn("INTERESTING", p)
        self.assertIn("manipulating", p.lower())
        self.assertIn("picks up a pan", p)
        self.assertIn("JSON array", p)


class TestParseVerdicts(unittest.TestCase):
    def test_plain_array(self):
        v = parse_verdicts('[{"i":0,"interesting":false,"reason":"talk"},{"i":1,"interesting":true,"reason":"pan"}]')
        self.assertFalse(v[0]["interesting"])
        self.assertTrue(v[1]["interesting"])

    def test_code_fenced_with_prose(self):
        text = 'Here are my labels:\n```json\n[{"i":0,"interesting":true,"reason":"x"}]\n```\nDone.'
        v = parse_verdicts(text)
        self.assertTrue(v[0]["interesting"])

    def test_malformed_returns_empty(self):
        self.assertEqual(parse_verdicts("not json at all"), {})


class TestFilterMap(unittest.TestCase):
    def test_build_and_default_uninteresting(self):
        rec = export_annotation_spans(_bundle(), "take1")
        v = {1: {"interesting": True, "reason": "pan"}}  # span 0 unlabeled
        fmap = build_filter_map(rec, v)
        self.assertEqual(len(fmap["regions"]), 2)
        self.assertFalse(fmap["regions"][0]["interesting"])  # unlabeled -> default not
        self.assertTrue(fmap["regions"][1]["interesting"])

    def test_regions_carry_channel(self):
        rec = export_annotation_spans(_bundle(), "take1")
        fmap = build_filter_map(rec, {1: {"interesting": True, "reason": "pan"}})
        for r in fmap["regions"]:
            self.assertEqual(r["channel"], "atomic_action")  # channel preserved for gating


class TestChannelAwareFilter(unittest.TestCase):
    """The consume-side fix: each channel filters against ITS OWN interesting regions,
    so a coarse activity_summary region can't keep fine atomic_action idle spans."""

    def _channels(self):
        return [
            {"name": "activity_summary", "kind": "interval", "spans": [
                {"start_s": 0.0, "end_s": 30.0, "text": "sitting on sofa, sometimes handling objects"},
            ]},
            {"name": "atomic_action", "kind": "interval", "spans": [
                {"start_s": 2.0, "end_s": 7.0, "text": "C is sitting still, listening"},   # idle
                {"start_s": 7.0, "end_s": 12.0, "text": "C grabs the cup and pours water"}, # manip
            ]},
        ]

    def test_coarse_interesting_does_not_keep_fine_idle(self):
        from src.gaze.training import _filter_channels_interesting
        # activity_summary[0..30] interesting; atomic span1 (grab/pour) interesting; atomic span0 (idle) NOT
        imap = {"regions": [
            {"start_s": 0.0, "end_s": 30.0, "channel": "activity_summary", "interesting": True},
            {"start_s": 2.0, "end_s": 7.0, "channel": "atomic_action", "interesting": False},
            {"start_s": 7.0, "end_s": 12.0, "channel": "atomic_action", "interesting": True},
        ]}
        out = _filter_channels_interesting(self._channels(), imap)
        atomic = next(c for c in out if c["name"] == "atomic_action")
        texts = [s["text"] for s in atomic["spans"]]
        # the idle atomic span is DROPPED even though it overlaps the interesting activity region
        self.assertNotIn("C is sitting still, listening", texts)
        self.assertIn("C grabs the cup and pours water", texts)

    def test_legacy_map_without_channel_falls_back(self):
        from src.gaze.training import _filter_channels_interesting
        imap = {"regions": [{"start_s": 0.0, "end_s": 30.0, "interesting": True}]}  # no channel
        out = _filter_channels_interesting(self._channels(), imap)
        # back-compat: any-channel overlap keeps both atomic spans
        atomic = next(c for c in out if c["name"] == "atomic_action")
        self.assertEqual(len(atomic["spans"]), 2)


if __name__ == "__main__":
    unittest.main()
