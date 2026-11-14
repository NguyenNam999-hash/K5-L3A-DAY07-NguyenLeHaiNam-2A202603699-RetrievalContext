"""Run the advanced agent on the advanced benchmark."""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.agent import AgentConfig, run_agent  # noqa: E402
from src.config import load_settings  # noqa: E402
from src.embeddings import FastEmbedder  # noqa: E402
from src.llm import LLMClient  # noqa: E402
from advanced.retrieval import build_advanced_retriever, load_chunks, ADVANCED_CONFIGS  # noqa: E402
from advanced.store import AdvancedPgStore  # noqa: E402

CONFIGS = {
    "baseline": {
        "retrieval": "dense",
        "agent": AgentConfig(
            name="advanced_baseline",
            max_searches=3,
            max_steps=8,
            default_top_k=4,
            budget=8000,
            enable_neighbors=False,
            enable_filter=False,
            enable_content_type=False,
            enable_get_page=False,
            include_sections=False,
            prompt_variant="default",
        ),
    },
    "selected": {
        "retrieval": "hybrid_rerank",
        "agent": AgentConfig(
            name="advanced_selected",
            max_searches=6,
            max_steps=14,
            default_top_k=8,
            budget=14000,
            enable_neighbors=False,
            enable_filter=False,
            enable_content_type=True,
            enable_get_page=True,
            include_sections=False,
            prompt_variant="advanced",
        ),
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--config", choices=tuple(CONFIGS), default="selected")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "advanced" / "results" / "agent")
    args = parser.parse_args()
    settings = load_settings(require_llm=True)
    questions = [
        question
        for question in json.loads((ROOT / "advanced" / "benchmark" / "questions.json").read_text(encoding="utf-8"))
        if question["split"] == args.split
    ]
    if args.limit:
        questions = questions[: args.limit]
    chunks = load_chunks()
    embedder = FastEmbedder(settings.embedding_model)
    store = AdvancedPgStore(settings.database_url, embedder.dimensions)
    if store.count() != len(chunks):
        raise SystemExit(f"index has {store.count()} chunks, file has {len(chunks)}; rebuild the index")
    llm = LLMClient(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.model,
        session_id=settings.session_id,
    )
    spec = CONFIGS[args.config]
    retriever = build_advanced_retriever(
        spec["retrieval"], embedder=embedder, store=store, chunks=chunks, llm=llm
    )
    agent_config: AgentConfig = spec["agent"]
    traces = []
    started = time.perf_counter()
    for index, question in enumerate(questions, start=1):
        print(f"[{index}/{len(questions)}] {question['id']} ({question['group']}): {question['question'][:70]}")
        try:
            trace = run_agent(
                question["question"],
                llm=llm,
                retriever=retriever,
                config=agent_config,
                question_id=question["id"],
                split=question["split"],
            )
        except Exception as exc:
            trace = {
                "query_id": question["id"],
                "split": question["split"],
                "group": question["group"],
                "question": question["question"],
                "status": "error",
                "answer": "",
                "citations": [],
                "events": [],
                "context": "",
                "context_ids": [],
                "context_chunks": [],
                "tool_calls": 0,
                "searches": 0,
                "latency_ms": 0,
                "llm_calls": 0,
                "usage": {},
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "config": args.config,
            }
        trace["group"] = question["group"]
        if question.get("answerable", True):
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
            "group": question["group"],
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
        "groups": {
            group: sum(1 for question in questions if question["group"] == group)
            for group in sorted({question["group"] for question in questions})
        },
        "retrieval": ADVANCED_CONFIGS[spec["retrieval"]].__dict__,
        "agent": {
            "model": settings.model,
            "max_searches": agent_config.max_searches,
            "max_steps": agent_config.max_steps,
            "top_k": agent_config.default_top_k,
            "budget": agent_config.budget,
            "content_type_filter": agent_config.enable_content_type,
            "get_page": agent_config.enable_get_page,
        },
        "index_chunks": len(chunks),
        "wall_seconds": round(time.perf_counter() - started, 3),
        "llm": llm.as_json(),
        "statuses": {
            status: sum(1 for trace in traces if trace["status"] == status)
            for status in ("answered", "abstain", "error")
        },
        "tool_calls_total": sum(int(trace.get("tool_calls") or 0) for trace in traces),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output_dir), **metadata["statuses"]}, indent=2))


if __name__ == "__main__":
    main()
