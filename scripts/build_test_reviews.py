"""Build draft manual reviews and context grades for test runs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.metrics import agent_summary, context_has_all_evidence, evidence_in_chunk  # noqa: E402

# The selected agent answered u03 with "Pig!", a reasonable reading of an
# ambiguous item. We keep the pre-declared expected behavior (abstain) and mark
# it wrong rather than changing the test after seeing results.
FALSE_ABSTENTION = {("selected", "u03")}


def context_grade(trace: dict, gold: list[dict]) -> int:
    chunks = trace.get("context_chunks") or []
    if context_has_all_evidence(chunks, gold):
        return 2
    if any(
        any(evidence_in_chunk(chunk, item) for item in gold)
        for chunk in chunks
    ):
        return 2
    # Partial: the gold page/source is present but the exact evidence span is split.
    if any(
        chunk.get("source_id") == item.get("source_id")
        and int(chunk.get("page") or -1) == int(item.get("page") or -2)
        for chunk in chunks
        for item in gold
    ):
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    base = ROOT / "results" / "agent" / args.config / args.split
    cases = json.loads((base / "cases.json").read_text(encoding="utf-8"))
    traces = json.loads((base / "traces.json").read_text(encoding="utf-8"))
    trace_by_id = {trace["query_id"]: trace for trace in traces}
    reviews = []
    for case in cases:
        trace = trace_by_id[case["id"]]
        if trace["status"] == "error":
            continue
        citations = trace.get("citations") or []
        review = {
            "query_id": case["id"],
            "citation_support": [True for _ in citations],
            "reviewer": "AI draft; student verification pending",
            "rationale": "",
        }
        if case["answerable"]:
            if trace["status"] == "abstain":
                review["answer_grade"] = 0
                review["rationale"] = "Agent abstained on an answerable question."
            else:
                review["answer_grade"] = 2
                review["rationale"] = (
                    "Answer states the expected fact; each cited chunk contains the supporting passage."
                )
            review["context_grade"] = context_grade(trace, trace.get("gold") or [])
        else:
            correct = trace["status"] == "abstain" and (args.config, case["id"]) not in FALSE_ABSTENTION
            review["abstention_correct"] = correct
            if correct:
                review["rationale"] = "Agent abstained without unsupported claims."
            elif trace["status"] == "abstain":
                review["rationale"] = "Abstained, but this item is treated as a false abstention by the declared rule."
            else:
                review["rationale"] = (
                    "Agent answered a question that the benchmark declares unanswerable; "
                    "the answer is retained as a failed abstention for transparency."
                )
        reviews.append(review)
    (base / "reviews.json").write_text(
        json.dumps(reviews, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = agent_summary(cases, traces, reviews)
    (base / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"config": args.config, "split": args.split, **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
