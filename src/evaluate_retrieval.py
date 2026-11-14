"""Compare retrieval methods on the dev or test split and save full traces."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_settings  # noqa: E402
from src.embeddings import FastEmbedder  # noqa: E402
from src.llm import LLMClient  # noqa: E402
from src.metrics import context_has_all_evidence, result_hit, retrieval_summary  # noqa: E402
from src.retrieval import (  # noqa: E402
    BASE_CONFIGS,
    build_retriever,
    infer_section,
    pack_context,
)
from src.store import PgVectorStore  # noqa: E402


def load_chunks(path: Path) -> list[dict[str, Any]]:
    chunks = json.loads(path.read_text(encoding="utf-8"))
    for chunk in chunks:
        chunk.setdefault("metadata", {})
        chunk.setdefault("section", chunk["metadata"].get("section", ""))
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--configs", default="tfidf,dense,bm25,hybrid,dense_filter,hybrid_filter,hybrid_filter_rerank")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--chunks", type=Path, default=ROOT / "data" / "day6" / "chunks_c.json")
    parser.add_argument("--questions", type=Path, default=ROOT / "benchmark" / "questions.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "retrieval")
    args = parser.parse_args()

    settings = load_settings(require_llm=True)
    chunks = load_chunks(args.chunks)
    questions = [
        question
        for question in json.loads(args.questions.read_text(encoding="utf-8"))
        if question["split"] == args.split and question.get("answerable", True)
    ]
    if args.limit:
        questions = questions[: args.limit]
    if not questions:
        raise SystemExit("no answerable questions in this split")
    sections = sorted({chunk["metadata"].get("section", "") for chunk in chunks if chunk["metadata"].get("section")})

    embedder = FastEmbedder(settings.embedding_model)
    store = PgVectorStore(settings.database_url, embedder.dimensions)
    if store.count(strategy=settings.strategy) != len(chunks):
        raise SystemExit(
            f"chunk file has {len(chunks)} chunks but pgvector strategy {settings.strategy} has "
            f"{store.count(strategy=settings.strategy)}; re-export or check the database"
        )
    llm = LLMClient(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.model,
        session_id=settings.session_id,
    )

    config_names = [name.strip() for name in args.configs.split(",") if name.strip()]
    all_summaries: dict[str, Any] = {}
    for config_name in config_names:
        if config_name not in BASE_CONFIGS:
            raise SystemExit(f"unknown config: {config_name}")
        config = BASE_CONFIGS[config_name]
        retriever = build_retriever(
            config,
            embedder=embedder,
            store=store,
            strategy=settings.strategy,
            chunks=chunks,
            llm=llm,
        )
        traces: list[dict[str, Any]] = []
        for question in questions:
            section = infer_section(question["question"], sections) if config.filters else None
            filters = {"section": section} if section else None
            started = time.perf_counter()
            rows = retriever.search(
                question["question"],
                top_k=max(10, config.top_k),
                filters=filters,
            )
            elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
            top3 = rows[:3]
            first_hit_rank = next(
                (row["rank"] for row in rows if result_hit(row, question["gold"])),
                None,
            )
            context, context_ids = pack_context(rows, budget=config.budget)
            context_by_id = {row["id"]: row for row in rows}
            context_chunks = [context_by_id[chunk_id] for chunk_id in context_ids if chunk_id in context_by_id]
            rerank_usage = next((row.get("rerank_usage") for row in rows if row.get("rerank_usage")), None)
            list_cost = None
            if rerank_usage:
                list_cost = rerank_usage.get("list_cost_usd")
                if list_cost is None:
                    list_cost = rerank_usage.get("cost_usd")
                if list_cost is None:
                    opencode = rerank_usage.get("opencode") or {}
                    list_cost = opencode.get("list_cost") if isinstance(opencode, dict) else None
            traces.append(
                {
                    "query_id": question["id"],
                    "split": args.split,
                    "question": question["question"],
                    "config": config.name,
                    "description": config.description,
                    "filter": filters,
                    "inferred_section": section,
                    "expected_answer": question["expected_answer"],
                    "gold": question["gold"],
                    "results": rows,
                    "top3": top3,
                    "hit1": bool(top3 and result_hit(top3[0], question["gold"])),
                    "hit3": any(result_hit(row, question["gold"]) for row in top3),
                    "first_hit_rank": first_hit_rank,
                    "context": context,
                    "context_ids": context_ids,
                    "context_chunks": context_chunks,
                    "context_chars": len(context),
                    "context_hit": context_has_all_evidence(context_chunks, question["gold"]),
                    "elapsed_ms": elapsed_ms,
                    "llm_calls": 1 if rerank_usage else 0,
                    "list_cost_usd": float(list_cost or 0.0),
                }
            )
        summary = {
            "config": config.name,
            "description": config.description,
            "split": args.split,
            "top_k": config.top_k,
            "budget": config.budget,
            "filters": config.filters,
            "n": len(traces),
            **retrieval_summary(traces),
        }
        all_summaries[config.name] = summary
        config_dir = args.output / config.name / args.split
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / "traces.json").write_text(
            json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (config_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"{config.name:24} {args.split:4} Hit@1 {summary['hit1']}/{summary['n']} "
            f"Hit@3 {summary['hit3']}/{summary['n']} Context {summary['context_hit']}/{summary['n']} "
            f"median {summary['median_latency_ms']} ms"
        )

    summary_path = args.output / f"summary_{args.split}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "split": args.split,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "embedding_model": settings.embedding_model,
                "agent_model": settings.model,
                "chunks": len(chunks),
                "questions": len(questions),
                "configs": all_summaries,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
