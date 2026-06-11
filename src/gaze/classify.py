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
    "Judge EACH span by what THAT span's own text says the HANDS are doing. Posture "
    "(standing/sitting/kneeling/crouching/leaning/walking) is neutral -- it neither "
    "makes a span interesting nor disqualifies it. Look past the posture to the hand "
    "action.\n"
    "\n"
    "INTERESTING (mark true) IF the span describes the camera wearer's hands ACTIVELY "
    "manipulating a physical object: picking up / putting down / moving / placing / "
    "handling an object, USING or OPERATING a tool or instrument (e.g. measuring with a "
    "tape measure, using a screwdriver), cleaning, wiping, preparing/cutting/pouring "
    "food or drink, washing, assembling, opening/closing or searching "
    "drawers-cabinets-appliances, playing a board/table game. A tool/object being "
    "actively USED counts even if the person is standing or walking while doing it "
    "(e.g. 'picks up the tape measure and measures the door' = INTERESTING).\n"
    "\n"
    "NOT INTERESTING (mark false), even if an object is named or hands are near it:\n"
    "  - conversation: talking, chatting, listening, gesturing/gesticulating to a peer;\n"
    "  - idle hands: 'resting both hands', 'hands on lap/knees', arms at side, just "
    "looking around -- no object being acted on;\n"
    "  - pure locomotion with no hand action: walking/turning between areas;\n"
    "  - passive media: watching TV, or using a remote merely to browse/search/navigate "
    "a TV or screen;\n"
    "  - merely HOLDING a static object while idle/talking/watching with no further "
    "action (e.g. 'holding a glass while chatting', 'resting a hand on a pillow');\n"
    "  - SPORTS and physical exercise: playing or practising any sport (basketball, "
    "soccer, bouldering/climbing, dancing, working out, throwing/kicking/dribbling a "
    "ball, lifting weights) -- this is whole-body activity, not fine hand manipulation;\n"
    "  - OUTDOOR activities: anything taking place outdoors / outside / in a yard, "
    "garden, street, park, field, court, or trail (gardening, walking a dog, outdoor "
    "sports, etc.). We want indoor hands-on object manipulation only.\n"
    "\n"
    "Decision rule: ask 'is a hand actively doing something TO an object in this span?' "
    "If yes -> interesting (the surrounding posture/conversation is irrelevant). If the "
    "hands are only idle, gesturing, holding-without-acting, or driving a TV remote -> "
    "not interesting. For a mixed span, judge by the hand action: 'sits and chats, then "
    "grabs the remote to search the TV' -> NOT interesting (remote-for-browsing); "
    "'sits on the sofa and folds the laundry' -> INTERESTING (active manipulation)."
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

    Output: ``{"regions": [{start_s, end_s, channel, interesting, reason}, ...]}`` on
    the same clock the spans carried. The ``channel`` is preserved so the consume side
    can gate each annotation channel against ITS OWN channel's verdicts (a coarse
    ``activity_summary`` region must not keep fine ``atomic_action`` idle sub-spans --
    that cross-granularity bleed was the conversational-leak source). Spans missing a
    verdict default to NOT interesting (conservative -- we'd rather drop an unlabeled
    region than train on noise).
    """
    regions: list[dict[str, Any]] = []
    for s in record["spans"]:
        v = verdicts.get(s["i"], {"interesting": False, "reason": "unlabeled"})
        if s.get("start_s") is None or s.get("end_s") is None:
            continue
        regions.append({
            "start_s": s["start_s"],
            "end_s": s["end_s"],
            "channel": s.get("channel"),
            "interesting": bool(v["interesting"]),
            "reason": v.get("reason", ""),
        })
    return {"take_id": record["take_id"], "regions": regions}


def _extract_json_array(text: str) -> list[Any]:
    text = text.strip()
    # strip code fences if present
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    start = text.find("[")
    if start == -1:
        return []
    end = text.rfind("]")
    if end != -1 and end > start:
        try:
            val = json.loads(text[start : end + 1])
            if isinstance(val, list):
                return val
        except json.JSONDecodeError:
            pass
    # Salvage a TRUNCATED array (e.g. the model hit max_tokens mid-array): parse every
    # complete top-level {...} object after the opening '['. Without this, a truncated
    # response silently parses to [] -> every span defaults to NOT interesting, which
    # is exactly the failure mode that zeroed out long nymeria takes.
    return _parse_objects(text[start + 1:])


def _parse_objects(body: str) -> list[Any]:
    """Parse each complete top-level JSON object in ``body`` (truncation-tolerant)."""
    out: list[Any] = []
    depth = 0
    in_str = False
    esc = False
    obj_start = None
    for i, ch in enumerate(body):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                obj_start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and obj_start is not None:
                try:
                    out.append(json.loads(body[obj_start : i + 1]))
                except json.JSONDecodeError:
                    pass
                obj_start = None
    return out
