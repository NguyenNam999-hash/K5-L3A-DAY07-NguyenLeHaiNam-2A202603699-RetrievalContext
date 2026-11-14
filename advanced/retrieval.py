"""Retrieval configuration for the advanced document."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.embeddings import Embedder
from src.retrieval import BM25Retriever, DenseRetriever, HybridRetriever, LLMReranker, RerankRetriever, Retriever
from advanced.store import AdvancedPgStore

ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = ROOT / "advanced" / "index" / "chunks.json"


@dataclass
class AdvancedConfig:
    name: str
    kind: str
    top_k: int
    budget: int
    description: str


ADVANCED_CONFIGS = {
    "dense": AdvancedConfig(
        "dense", "dense", top_k=6, budget=8000,
        description="BGE + pgvector over all text/table/chart/image chunks, top-6",
    ),
    "hybrid": AdvancedConfig(
        "hybrid", "hybrid", top_k=8, budget=12000,
        description="Dense + BM25 RRF over all content types, top-8",
    ),
    "hybrid_rerank": AdvancedConfig(
        "hybrid_rerank", "hybrid_rerank", top_k=8, budget=14000,
        description="Hybrid RRF top-24 + LLM listwise rerank, top-8",
    ),
}


def load_chunks(path: Path = CHUNKS_PATH) -> list[dict[str, Any]]:
    chunks = json.loads(path.read_text(encoding="utf-8"))
    for chunk in chunks:
        chunk.setdefault("metadata", {})
        chunk["metadata"].setdefault("content_type", chunk["content_type"])
    return chunks


def build_advanced_retriever(
    name: str,
    *,
    embedder: Embedder,
    store: AdvancedPgStore,
    chunks: list[dict[str, Any]],
    llm: Any | None = None,
) -> Retriever:
    config = ADVANCED_CONFIGS[name]
    dense = DenseRetriever(embedder, store, strategy=None, chunks=chunks, name="dense", budget=config.budget)
    lexical = BM25Retriever(chunks, name="bm25", budget=config.budget)
    if config.kind == "dense":
        dense.name = config.name
        return dense
    hybrid = HybridRetriever(dense, lexical, name=config.name, budget=config.budget, candidate_k=32)
    if config.kind == "hybrid":
        return hybrid
    if llm is None:
        raise ValueError("LLM required for hybrid_rerank")
    return RerankRetriever(
        hybrid,
        LLMReranker(llm, candidate_chars=1000),
        candidate_k=24,
        name=config.name,
        budget=config.budget,
    )
