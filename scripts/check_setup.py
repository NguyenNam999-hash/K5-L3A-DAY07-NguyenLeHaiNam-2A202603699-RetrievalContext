"""Check Day 6 pgvector, the embedding model, and the agent endpoint."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import load_settings  # noqa: E402
from src.embeddings import FastEmbedder  # noqa: E402
from src.llm import LLMClient  # noqa: E402
from src.store import PgVectorStore  # noqa: E402


def main() -> None:
    settings = load_settings(require_llm=True)
    embedder = FastEmbedder(settings.embedding_model)
    store = PgVectorStore(settings.database_url, embedder.dimensions)
    counts = {strategy: store.count(strategy=strategy) for strategy in ("A", "B", "C")}
    vector = embedder.embed_query("Alice and the golden key")
    hits = store.search(vector, top_k=1, strategy=settings.strategy)
    llm = LLMClient(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.model,
        session_id=settings.session_id,
    )
    response = llm.chat(
        [{"role": "user", "content": "Reply with the single word OK."}],
        max_tokens=50,
        temperature=0,
    )
    print(
        json.dumps(
            {
                "database": "ok",
                "strategy": settings.strategy,
                "chunk_counts": counts,
                "embedding_model": embedder.model_name,
                "dimensions": embedder.dimensions,
                "top1": hits[0]["id"] if hits else None,
                "agent_model": settings.model,
                "agent_finish_reason": response["finish_reason"],
                "agent_content": (response["message"].get("content") or "")[:80],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
