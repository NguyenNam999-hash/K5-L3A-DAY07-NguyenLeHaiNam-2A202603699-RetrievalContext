"""Export Day 6 strategy C chunks into the Day 7 JSON contract."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import load_settings  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", default="C")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "day6")
    args = parser.parse_args()
    settings = load_settings(require_llm=False)
    import psycopg

    with psycopg.connect(settings.database_url, connect_timeout=10) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT chunk_id, document_id, source, page, text, metadata
                FROM chunks WHERE strategy = %s ORDER BY chunk_id
                """,
                (args.strategy,),
            )
            rows = cursor.fetchall()
    if not rows:
        raise SystemExit(f"No chunks found for strategy {args.strategy!r}.")
    chunks = []
    for chunk_id, document_id, source, page, text, metadata in rows:
        metadata = metadata if isinstance(metadata, dict) else json.loads(metadata or "{}")
        chunks.append(
            {
                "id": chunk_id,
                "text": text,
                "source_id": document_id,
                "source": source,
                "page": int(page),
                "metadata": metadata,
            }
        )
    if len({chunk["id"] for chunk in chunks}) != len(chunks):
        raise SystemExit("Duplicate chunk IDs in the Day 6 export.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "chunks_c.json").write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    day6_manifest_path = settings.day6_root / "data" / "manifest.json"
    manifest = json.loads(day6_manifest_path.read_text(encoding="utf-8")) if day6_manifest_path.exists() else {}
    export_manifest = {
        "source_id": manifest.get("source_id"),
        "title": manifest.get("title"),
        "author": manifest.get("author"),
        "source_path_or_url": manifest.get("source_path_or_url"),
        "license_or_permission": manifest.get("license_or_permission"),
        "accessed_at": manifest.get("accessed_at"),
        "snapshot": manifest.get("snapshot"),
        "raw_sha256": manifest.get("raw_sha256"),
        "page_count": manifest.get("page_count"),
        "content_chars": manifest.get("content_chars"),
        "day6_root": str(settings.day6_root),
        "strategy": args.strategy,
        "strategy_description": "B paragraph chunks <=1000 chars; C adds section metadata filter",
        "embedding_model": settings.embedding_model,
        "vector_store": "PostgreSQL + pgvector, table chunks, cosine exact scan",
        "chunk_count": len(chunks),
        "pages": manifest.get("pages", []),
    }
    (args.output_dir / "source_manifest.json").write_text(
        json.dumps(export_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Exported {len(chunks)} strategy-{args.strategy} chunks to {args.output_dir}")


if __name__ == "__main__":
    main()
