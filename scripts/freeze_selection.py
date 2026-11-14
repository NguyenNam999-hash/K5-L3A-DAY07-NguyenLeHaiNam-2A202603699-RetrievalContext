"""Freeze the selected configuration on dev before any test-split run."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmark" / "questions.json"


def main() -> None:
    retrieval = json.loads((ROOT / "results" / "retrieval" / "summary_dev.json").read_text(encoding="utf-8"))
    agent_metrics = {
        config: json.loads(
            (ROOT / "results" / "agent" / config / "dev" / "metrics.json").read_text(encoding="utf-8")
        )
        for config in ("baseline", "hybrid_filter", "selected")
    }
    test_ids = [
        question["id"]
        for question in json.loads(BENCHMARK.read_text(encoding="utf-8"))
        if question["split"] == "test"
    ]
    selection = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "criteria": [
            "1. Evidence Hit@3 on dev (primary retrieval quality)",
            "2. Context Evidence Rate on dev",
            "3. Grounded answer mean on dev (manual labels)",
            "4. Latency and list cost as tie-breakers",
        ],
        "chosen_retrieval": "hybrid_filter_rerank",
        "chosen_agent": "selected",
        "rationale": (
            "Hybrid + section filter + LLM rerank has the highest dev Evidence Hit@3 (15/15), "
            "the highest Hit@1 (15/15), and the highest Context Evidence Rate (15/15). "
            "The selected agent also reaches grounded-answer mean 2.00 with zero false "
            "abstentions and correct abstention on both dev unanswerable questions. "
            "Hybrid+filter is much cheaper (29 ms vs 6.7 s median retrieval) but loses one "
            "dev Hit@3, so the published primary criterion selects rerank."
        ),
        "dev_retrieval": retrieval["configs"],
        "dev_agent": agent_metrics,
        "test_split_frozen": True,
        "test_ids": test_ids,
        "test_benchmark_sha256": hashlib.sha256(BENCHMARK.read_bytes()).hexdigest(),
        "test_results_seen": False,
    }
    path = ROOT / "results" / "selection.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Froze selection to {path}")
    print(f"Test questions: {test_ids}")
    print(f"Benchmark sha256: {selection['test_benchmark_sha256']}")


if __name__ == "__main__":
    main()
