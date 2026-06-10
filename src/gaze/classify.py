"""Interesting-region classification for nymeria (and any dataset) annotations.

Long nymeria takes contain extended uninteresting stretches (static, conversation).
For training we want only INTERESTING regions:

    interesting  = manipulation of objects, cleaning, playing board games,
                   object interaction; LOCOMOTION is acceptable.
    NOT interesting = conversational periods, static/idle periods.

This module:
  * exports each take's annotation spans (text + times) for classification
    (``export_annotation_spans``),
  * builds the classification PROMPT for a batch of spans (``classify_prompt``)
    and parses the model's JSON verdicts (``parse_verdicts``),
  * assembles a per-take filter map (``build_filter_map``) of the form
    ``{take_id: {"regions": [{start_s, end_s, interesting, reason}, ...]}}``
    consumed by ``training._intersect_interesting``.

The actual model call is done by the caller: Claude subagents inline for the
smoke set, or ``scripts/classify_nymeria_regions.py`` (Bedrock) for the full
dataset. This module is pure/stdlib so it is unit-testable without a model.
"""
from __future__ import annotations

import json
import re
from typing import Any


INTERESTING_RUBRIC = (
    "INTERESTING = the camera wearer is actively MANIPULATING objects with their "
    "hands: picking up / putting down / handling / using objects, cleaning, "
    "preparing food, playing board/table games, opening/closing or searching "
    "through drawers-cabinets-appliances, assembling/operating tools. There must be "
    "a hands-on object-manipulation component.\n"
    "NOT INTERESTING (exclude even if other things happen in the span):\n"
    "  - conversational periods (talking/gesturing to a peer with no hands-on manipulation),\n"
    "  - static/idle periods (standing or sitting still, just looking around),\n"
    "  - PURE locomotion with no manipulation (walking/moving between areas, even if "
    "the person then sits or talks),\n"
    "  - passive watching (e.g. watching TV, or using a remote merely to browse/watch).\n"
    "A span that is mostly walking-then-talking, or sitting-and-chatting, or "
    "searching-for-a-movie-on-TV is NOT interesting. Only mark interesting when there "
    "is genuine hands-on manipulation of physical objects."
)


def export_annotation_spans(bundle: dict[str, Any], take_id: str) -> dict[str, Any]:
    """Flatten one take's annotation channels into a classification-ready record.

    ``bundle`` is the extracted ``<slug>_full.json`` dict (has ``annotations`` =
    list of channels with ``segments``). Returns ``{take_id, spans:[{i,start_s,
    end_s,channel,text}]}`` with raw (un-reconciled) times -- the classifier only
    needs the text + relative ordering; the filter map is matched back to the
    reconciled clock at consume time via the same start/end values.
    """
    spans: list[dict[str, Any]] = []
    i = 0
    for ch in bundle.get("annotations", []) or []:
        for s in ch.get("segments", []) or []:
            text = s.get("text")
            if not text or str(text).strip() == "":
                continue
            spans.append({
                "i": i,
                "start_s": s.get("start_s"),
                "end_s": s.get("end_s"),
                "channel": ch.get("name"),
                "text": str(text),
            })
            i += 1
    return {"take_id": take_id, "spans": spans}


def classify_prompt(record: dict[str, Any], rubric: str = INTERESTING_RUBRIC) -> str:
    """Build the LLM prompt to classify each span of one take as interesting or not."""
    lines = [
        "You are labeling egocentric-video annotation spans as INTERESTING or not "
        "for a robot-manipulation gaze-training dataset.",
        "",
        rubric,
        "",
        f"Take: {record['take_id']}",
        "Spans (index | seconds | text):",
    ]
    for s in record["spans"]:
        a = s.get("start_s"); b = s.get("end_s")
        rng = f"{a:.1f}-{b:.1f}" if a is not None and b is not None else "?"
        lines.append(f"  {s['i']} | {rng} | {s['text']}")
    lines += [
        "",
        "Return ONLY a JSON array, one object per span index, like:",
        '[{"i": 0, "interesting": true, "reason": "manipulating a pan"}, ...]',
        "Every span index must appear exactly once.",
    ]
    return "\n".join(lines)


def parse_verdicts(text: str) -> dict[int, dict[str, Any]]:
    """Parse a model response into ``{span_index: {interesting, reason}}``.

    Tolerant of prose around the JSON array (extracts the first top-level array).
    """
    arr = _extract_json_array(text)
    out: dict[int, dict[str, Any]] = {}
    for item in arr:
        if not isinstance(item, dict) or "i" not in item:
            continue
        out[int(item["i"])] = {
            "interesting": bool(item.get("interesting")),
            "reason": str(item.get("reason", "")),
        }
    return out


def build_filter_map(record: dict[str, Any], verdicts: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Combine a take's spans with per-span verdicts into the consume-side map.

    Output: ``{"regions": [{start_s, end_s, interesting, reason}, ...]}`` on the
    same clock the spans carried. Spans missing a verdict default to NOT interesting
    (conservative -- we'd rather drop an unlabeled region than train on noise).
    """
    regions: list[dict[str, Any]] = []
    for s in record["spans"]:
        v = verdicts.get(s["i"], {"interesting": False, "reason": "unlabeled"})
        if s.get("start_s") is None or s.get("end_s") is None:
            continue
        regions.append({
            "start_s": s["start_s"],
            "end_s": s["end_s"],
            "interesting": bool(v["interesting"]),
            "reason": v.get("reason", ""),
        })
    return {"take_id": record["take_id"], "regions": regions}


def _extract_json_array(text: str) -> list[Any]:
    text = text.strip()
    # strip code fences if present
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        return []
    try:
        val = json.loads(text[start : end + 1])
        return val if isinstance(val, list) else []
    except json.JSONDecodeError:
        return []
