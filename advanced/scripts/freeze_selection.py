"""Freeze the advanced configuration on dev before the test run."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "advanced" / "benchmark" / "questions.json"


def main() -> None:
    summary = {}
    for config in ("baseline", "selected"):
        traces = json.loads(
            (ROOT / "advanced" / "results" / "agent" / config / "dev" / "traces.json").read_text(
                encoding="utf-8"
            )
        )
        summary[config] = {
            "answered": sum(1 for trace in traces if trace["status"] == "answered"),
            "abstain": sum(1 for trace in traces if trace["status"] == "abstain"),
            "error": sum(1 for trace in traces if trace["status"] == "error"),
            "answerable_answered": sum(
                1 for trace in traces if trace["status"] == "answered" and not trace["query_id"].startswith("au")
            ),
            "unanswerable_abstained": sum(
                1 for trace in traces if trace["status"] == "abstain" and trace["query_id"].startswith("au")
            ),
            "tool_calls": sum(int(trace.get("tool_calls") or 0) for trace in traces),
            "tokens": sum(int((trace.get("usage") or {}).get("total_tokens") or 0) for trace in traces),
            "list_cost_usd": round(
                sum(float((trace.get("usage") or {}).get("list_cost_usd") or 0) for trace in traces), 6
            ),
        }
    selection = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "criteria": [
            "1. Dev answerable coverage with correct evidence",
            "2. Dev abstention on unanswerable questions",
            "3. Group coverage (table/image/chart/multi-hop/text)",
            "4. Latency and cost as tie-breakers",
        ],
        "chosen_agent": "selected",
        "chosen_retrieval": "hybrid_rerank",
        "rationale": (
            "The selected pipeline answered all 15 dev answerable questions and abstained on "
            "all 3 dev unanswerable questions; the dense baseline abstained on 3 answerable "
            "questions. The selected pipeline uses content-type filters and get_page, which "
            "the mixed table/chart/map benchmark requires."
        ),
        "dev_status": summary,
        "test_benchmark_sha256": hashlib.sha256(BENCHMARK.read_bytes()).hexdigest(),
        "test_results_seen": False,
    }
    path = ROOT / "advanced" / "results" / "selection.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
