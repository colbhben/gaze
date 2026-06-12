#!/usr/bin/env python3
"""Classify nymeria (or any dataset) annotation regions as interesting via Claude on Bedrock.

FULL-DATASET path (deferred to train/val prep). For the SMOKE set, classification is
done inline with Claude subagents instead (no batch infra needed).

Pipeline:
  1. `gaze curate export-annotations --dataset nymeria --episodes-file all_takes.json
        --out /tmp/nymeria_anno.jsonl`  -> one record per take.
  2. THIS script: for each record, build the classify prompt (gaze.classify.classify_prompt),
     call Claude on Bedrock, parse verdicts (parse_verdicts), build the filter map
     (build_filter_map), and write {take_id: {regions:[...]}} to --out.
  3. `gaze curate build-training-manifest --interesting-map nymeria=/tmp/nymeria_map.json`.

Bedrock is selected because CLAUDE_CODE_USE_BEDROCK is set in this environment. Requires
boto3 + AWS creds with bedrock:InvokeModel. Run with --dry-run to preview prompts/cost
without calling the model.

Usage:
  python scripts/classify_nymeria_regions.py --in /tmp/nymeria_anno.jsonl \
      --out /tmp/nymeria_map.json [--model <bedrock-model-id>] [--dry-run] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gaze.classify import classify_prompt, parse_verdicts, build_filter_map

DEFAULT_MODEL = os.environ.get(
    # Opus 4.8: best accuracy on the subtle 'completed manipulation vs looking / waiting
    # / incomplete / peer-game' distinctions in the rubric. Override with --model or
    # GAZE_CLASSIFY_MODEL (e.g. us.anthropic.claude-sonnet-4-6 for a cheaper/faster pass).
    "GAZE_CLASSIFY_MODEL", "us.anthropic.claude-opus-4-8"
)


# Spans per model call. Each verdict (with a short reason) is ~40-70 output tokens, so
# a chunk of 60 spans stays well under max_tokens and never truncates mid-array (the
# bug that silently zeroed long takes). Large takes are split into ceil(N/60) chunks.
SPANS_PER_CHUNK = 60
MAX_TOKENS = 8192


def call_bedrock(prompt: str, model_id: str, region: str | None = None, max_tokens: int = MAX_TOKENS) -> tuple[str, str | None]:
    """Invoke a Claude model on Bedrock; return (text, stop_reason)."""
    import boto3  # lazy: only needed for the real run

    client = boto3.client("bedrock-runtime", region_name=region or os.environ.get("AWS_REGION", "us-east-1"))
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
    }
    resp = client.invoke_model(modelId=model_id, body=json.dumps(body))
    payload = json.loads(resp["body"].read())
    text = "".join(block.get("text", "") for block in payload.get("content", []))
    return text, payload.get("stop_reason")


def _chunk_record(rec: dict, size: int = SPANS_PER_CHUNK) -> list[dict]:
    """Split a take into <=size-span sub-records, preserving original span indices."""
    spans = rec["spans"]
    return [{"take_id": rec["take_id"], "spans": spans[i : i + size]}
            for i in range(0, len(spans), size)] or [rec]


def classify_record(rec: dict, model: str, region: str | None) -> dict[int, dict]:
    """Classify ALL spans of one take, chunked so no response truncates. Re-splits a
    chunk that still hits max_tokens (defensive). Returns merged {span_index: verdict}."""
    verdicts: dict[int, dict] = {}
    pending = _chunk_record(rec)
    while pending:
        chunk = pending.pop(0)
        text, stop = call_bedrock(classify_prompt(chunk), model, region=region)
        if stop == "max_tokens" and len(chunk["spans"]) > 1:
            mid = len(chunk["spans"]) // 2
            pending.insert(0, {"take_id": chunk["take_id"], "spans": chunk["spans"][mid:]})
            pending.insert(0, {"take_id": chunk["take_id"], "spans": chunk["spans"][:mid]})
            continue
        verdicts.update(parse_verdicts(text))
    return verdicts


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True, help="export-annotations JSONL")
    ap.add_argument("--out", required=True, help="filter-map JSON {take_id:{regions:[...]}}")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"Bedrock model id; default {DEFAULT_MODEL}")
    ap.add_argument("--region", default=None, help="AWS region")
    ap.add_argument("--limit", type=int, default=None, help="only classify the first N takes")
    ap.add_argument("--workers", type=int, default=8, help="parallel takes; default 8")
    ap.add_argument("--resume", action="store_true", help="reuse takes already present in --out (and any --reuse-map); only classify the rest. Crash-safe / append.")
    ap.add_argument("--reuse-map", action="append", default=[], help="existing interesting-map JSON(s) to seed from; takes present there are NOT re-classified (repeatable).")
    ap.add_argument("--dry-run", action="store_true", help="print prompts + cost estimate, no model call")
    args = ap.parse_args(argv)

    records = [json.loads(line) for line in Path(args.inp).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        records = records[: args.limit]

    # RESUME / REUSE: seed from prior maps (the preserved interesting-list is reusable here)
    # so already-classified takes are never re-paid for. Existing verdicts pass through.
    out_map: dict[str, dict] = {}
    for mp in (args.reuse_map + ([args.out] if args.resume else [])):
        if mp and Path(mp).exists():
            try:
                prior = json.loads(Path(mp).read_text(encoding="utf-8"))
                out_map.update(prior)
            except (json.JSONDecodeError, OSError):
                pass
    if out_map:
        before = len(records)
        records = [r for r in records if r["take_id"] not in out_map]
        print(f"[resume] seeded {len(out_map)} takes from prior map(s); {len(records)}/{before} left to classify", flush=True)

    if args.dry_run:
        total_spans = sum(len(r["spans"]) for r in records)
        total_chunks = sum(len(_chunk_record(r)) for r in records)
        print(f"[dry-run] {len(records)} takes, {total_spans} spans, {total_chunks} model calls; model={args.model}")
        if records:
            print("--- sample prompt (take 0, chunk 0) ---")
            print(classify_prompt(_chunk_record(records[0])[0])[:1400])
        return 0

    import concurrent.futures
    done = 0

    def work(rec):
        try:
            verdicts = classify_record(rec, args.model, args.region)
        except Exception as exc:  # noqa
            print(f"warn: take {rec['take_id']} failed: {exc}", file=sys.stderr)
            verdicts = {}
        return rec, build_filter_map(rec, verdicts)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        for rec, fmap in ex.map(work, records):
            out_map[rec["take_id"]] = {"regions": fmap["regions"]}
            done += 1
            n_int = sum(1 for r in fmap["regions"] if r["interesting"])
            print(f"[{done}/{len(records)}] {rec['take_id']}: {n_int}/{len(fmap['regions'])} interesting", flush=True)

    Path(args.out).write_text(json.dumps(out_map, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {len(out_map)} takes -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
