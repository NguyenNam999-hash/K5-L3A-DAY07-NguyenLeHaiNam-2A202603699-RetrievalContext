"""Embed and index the advanced chunks into pgvector."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.config import load_settings  # noqa: E402
from src.embeddings import FastEmbedder  # noqa: E402
from advanced.store import AdvancedPgStore  # noqa: E402


def main() -> None:
    settings = load_settings(require_llm=False)
    chunks_path = ROOT / "advanced" / "index" / "chunks.json"
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    if not chunks:
        raise SystemExit("no chunks; run advanced/scripts/extract_document.py first")
    embedder = FastEmbedder(settings.embedding_model)
    store = AdvancedPgStore(settings.database_url, embedder.dimensions)
    store.create_schema()
    started = time.perf_counter()
    vectors = embedder.embed_documents([chunk["text"] for chunk in chunks])
    embedding_seconds = round(time.perf_counter() - started, 2)
    store.upsert(chunks, vectors)
    metadata = {
        "embedding_model": settings.embedding_model,
        "dimensions": embedder.dimensions,
        "chunks": len(chunks),
        "counts_by_type": {
            kind: sum(1 for chunk in chunks if chunk["content_type"] == kind)
            for kind in ("text", "table", "chart", "image")
        },
        "embedding_seconds": embedding_seconds,
        "store": "PostgreSQL + pgvector, table advanced_chunks, cosine exact scan",
        "indexed_count": store.count(),
    }
    (ROOT / "advanced" / "index" / "index_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
