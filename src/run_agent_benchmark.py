"""Run the real agent on a benchmark split and save traces for manual review."""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent import AgentConfig, run_agent  # noqa: E402
from src.config import load_settings  # noqa: E402
from src.embeddings import FastEmbedder  # noqa: E402
from src.llm import LLMClient  # noqa: E402
from src.retrieval import BASE_CONFIGS, build_retriever  # noqa: E402
from src.store import PgVectorStore  # noqa: E402

AGENT_CONFIGS = {
    "baseline": {
        "retrieval": "dense",
        "agent": AgentConfig(
            name="baseline",
            max_searches=2,
            max_steps=5,
            default_top_k=3,
            budget=2400,
            enable_neighbors=False,
            enable_filter=False,
            prompt_variant="baseline",
        ),
    },
    "hybrid_filter": {
        "retrieval": "hybrid_filter",
        "agent": AgentConfig(
            name="hybrid_filter",
            max_searches=4,
            max_steps=8,
            default_top_k=5,
            budget=4800,
            enable_neighbors=True,
            enable_filter=True,
        ),
    },
    "selected": {
        "retrieval": "hybrid_filter_rerank",
        "agent": AgentConfig(
            name="selected",
            max_searches=5,
            max_steps=10,
            default_top_k=6,
            budget=6000,
            enable_neighbors=True,
            enable_filter=True,
        ),
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--config", choices=tuple(AGENT_CONFIGS), default="baseline")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "agent")
    args = parser.parse_args()

    settings = load_settings(require_llm=True)
    chunks = json.loads((ROOT / "data" / "day6" / "chunks_c.json").read_text(encoding="utf-8"))
    questions = [
        question
        for question in json.loads((ROOT / "benchmark" / "questions.json").read_text(encoding="utf-8"))
        if question["split"] == args.split
    ]
    if args.limit:
        questions = questions[: args.limit]
    if not questions:
        raise SystemExit("no questions in this split")

    spec = AGENT_CONFIGS[args.config]
    retrieval_config = BASE_CONFIGS[spec["retrieval"]]
    agent_config: AgentConfig = spec["agent"]
    embedder = FastEmbedder(settings.embedding_model)
    store = PgVectorStore(settings.database_url, embedder.dimensions)
    llm = LLMClient(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.model,
        session_id=settings.session_id,
    )
    retriever = build_retriever(
        retrieval_config,
        embedder=embedder,
        store=store,
        strategy=settings.strategy,
        chunks=chunks,
        llm=llm,
    )

    traces: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, question in enumerate(questions, start=1):
        print(f"[{index}/{len(questions)}] {question['id']}: {question['question'][:70]}")
        try:
            trace = run_agent(
                question["question"],
                llm=llm,
                retriever=retriever,
                config=agent_config,
                question_id=question["id"],
                split=question["split"],
            )
        except Exception as exc:  # recorded, not replaced with fake output
            trace = {
                "query_id": question["id"],
                "split": question["split"],
                "question": question["question"],
                "status": "error",
                "answer": "",
                "citations": [],
                "events": [],
                "context": "",
                "context_ids": [],
                "tool_calls": 0,
                "searches": 0,
                "latency_ms": 0,
                "llm_calls": 0,
                "usage": {},
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "config": args.config,
            }
        if question.get("answerable", True) and "gold" in question:
            trace["gold"] = question["gold"]
            trace["expected_answer"] = question["expected_answer"]
        traces.append(trace)

    output_dir = args.output / args.config / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "traces.json").write_text(
        json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    cases = [
        {
            "id": question["id"],
            "split": question["split"],
            "answerable": bool(question.get("answerable", True)),
        }
        for question in questions
    ]
    (output_dir / "cases.json").write_text(
        json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": args.config,
        "split": args.split,
        "questions": len(questions),
        "agent": {
            "model": settings.model,
            "max_searches": agent_config.max_searches,
            "max_steps": agent_config.max_steps,
            "top_k": agent_config.default_top_k,
            "budget": agent_config.budget,
            "enable_filter": agent_config.enable_filter,
            "enable_neighbors": agent_config.enable_neighbors,
        },
        "retrieval": {
            "name": retrieval_config.name,
            "description": retrieval_config.description,
            "top_k": retrieval_config.top_k,
            "budget": retrieval_config.budget,
        },
        "chunks": len(chunks),
        "wall_seconds": round(time.perf_counter() - started, 3),
        "llm": llm.as_json(),
        "errors": sum(1 for trace in traces if trace["status"] == "error"),
        "statuses": {
            status: sum(1 for trace in traces if trace["status"] == status)
            for status in ("answered", "abstain", "error")
        },
        "tool_calls_total": sum(int(trace.get("tool_calls") or 0) for trace in traces),
        "usage_total": llm.total_usage,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output_dir), **metadata["statuses"], "errors": metadata["errors"]}, indent=2))


if __name__ == "__main__":
    main()
