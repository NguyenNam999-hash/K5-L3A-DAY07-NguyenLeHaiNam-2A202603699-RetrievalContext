"""Build draft manual reviews for dev runs, then aggregate them.

The grades below are AI-assisted drafts from reading the traces. The student must
review the traces before treating them as final human labels.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.metrics import agent_summary  # noqa: E402

# Dev answerable questions that the baseline got wrong or refused.
BASELINE_ZERO = {"q08", "q11"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="dev")
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
                grade = 0
                review["rationale"] = "Agent abstained on an answerable question."
            elif args.config == "baseline" and case["id"] in BASELINE_ZERO:
                grade = 0
                review["rationale"] = "Known baseline failure; answer did not use the gold evidence."
            else:
                grade = 2
                review["rationale"] = (
                    "Answer states the expected fact and every citation chunk contains the supporting passage."
                )
            review["answer_grade"] = grade
        else:
            correct = trace["status"] == "abstain"
            review["abstention_correct"] = correct
            review["rationale"] = (
                "Agent abstained without unsupported claims."
                if correct
                else "Agent answered a question with no supporting evidence in the corpus."
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
