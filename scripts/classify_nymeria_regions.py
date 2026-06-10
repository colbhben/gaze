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
    "GAZE_CLASSIFY_MODEL", "us.anthropic.claude-sonnet-4-6"
)


def call_bedrock(prompt: str, model_id: str, region: str | None = None, max_tokens: int = 4096) -> str:
    """Invoke a Claude model on Bedrock and return the text response."""
    import boto3  # lazy: only needed for the real run

    client = boto3.client("bedrock-runtime", region_name=region or os.environ.get("AWS_REGION", "us-east-1"))
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
    }
    resp = client.invoke_model(modelId=model_id, body=json.dumps(body))
    payload = json.loads(resp["body"].read())
    return "".join(block.get("text", "") for block in payload.get("content", []))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True, help="export-annotations JSONL")
    ap.add_argument("--out", required=True, help="filter-map JSON {take_id:{regions:[...]}}")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"Bedrock model id; default {DEFAULT_MODEL}")
    ap.add_argument("--region", default=None, help="AWS region")
    ap.add_argument("--limit", type=int, default=None, help="only classify the first N takes")
    ap.add_argument("--dry-run", action="store_true", help="print prompts + cost estimate, no model call")
    args = ap.parse_args(argv)

    records = [json.loads(line) for line in Path(args.inp).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        records = records[: args.limit]

    if args.dry_run:
        total_spans = sum(len(r["spans"]) for r in records)
        print(f"[dry-run] {len(records)} takes, {total_spans} spans; model={args.model}")
        if records:
            print("--- sample prompt (take 0) ---")
            print(classify_prompt(records[0])[:1200])
        return 0

    out_map: dict[str, dict] = {}
    for k, rec in enumerate(records):
        prompt = classify_prompt(rec)
        try:
            text = call_bedrock(prompt, args.model, region=args.region)
            verdicts = parse_verdicts(text)
        except Exception as exc:  # noqa
            print(f"warn: take {rec['take_id']} failed: {exc}", file=sys.stderr)
            verdicts = {}
        fmap = build_filter_map(rec, verdicts)
        out_map[rec["take_id"]] = {"regions": fmap["regions"]}
        n_int = sum(1 for r in fmap["regions"] if r["interesting"])
        print(f"[{k + 1}/{len(records)}] {rec['take_id']}: {n_int}/{len(fmap['regions'])} interesting", flush=True)

    Path(args.out).write_text(json.dumps(out_map, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {len(out_map)} takes -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
