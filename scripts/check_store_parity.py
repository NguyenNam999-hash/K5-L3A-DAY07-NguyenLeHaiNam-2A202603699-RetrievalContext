"""Verify that the local snapshot reproduces pgvector top-k results."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import load_settings  # noqa: E402
from src.embeddings import FastEmbedder  # noqa: E402
from src.store import PgVectorStore, SnapshotStore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, default=ROOT / "data" / "day6" / "chunks_c.json")
    parser.add_argument("--vectors", type=Path, default=ROOT / "data" / "day6" / "vectors_c.npz")
    parser.add_argument("--questions", type=Path, default=ROOT / "benchmark" / "questions.json")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if not args.vectors.exists():
        raise SystemExit(f"missing {args.vectors}; run scripts/build_snapshot.py first")
    settings = load_settings(require_llm=False)
    embedder = FastEmbedder(settings.embedding_model)
    pg = PgVectorStore(settings.database_url, embedder.dimensions)
    snapshot = SnapshotStore(args.chunks, args.vectors, args.vectors.parent / "vectors_c_ids.json")
    questions = [
        question
        for question in json.loads(args.questions.read_text(encoding="utf-8"))
        if question["split"] == "dev" and question.get("answerable", True)
    ]
    if args.limit:
        questions = questions[: args.limit]
    mismatches = []
    for question in questions:
        vector = embedder.embed_query(question["question"])
        pg_ids = [row["id"] for row in pg.search(vector, top_k=args.top_k, strategy=settings.strategy)]
        snap_ids = [row["id"] for row in snapshot.search(vector, top_k=args.top_k)]
        if pg_ids != snap_ids:
            mismatches.append({"id": question["id"], "pgvector": pg_ids, "snapshot": snap_ids})
    print(json.dumps({"questions": len(questions), "top_k": args.top_k, "mismatches": mismatches}, indent=2))
    if mismatches:
        raise SystemExit("snapshot does not reproduce pgvector top-k")
    print("Store parity OK")


if __name__ == "__main__":
    main()
