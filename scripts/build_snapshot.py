"""Rebuild a local exact-cosine snapshot of the Day 6 strategy-C vectors."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import load_settings  # noqa: E402
from src.embeddings import FastEmbedder  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, default=ROOT / "data" / "day6" / "chunks_c.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "day6" / "vectors_c.npz")
    args = parser.parse_args()
    settings = load_settings(require_llm=False)
    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    embedder = FastEmbedder(settings.embedding_model)
    vectors = embedder.embed_documents([chunk["text"] for chunk in chunks])
    ids = [chunk["id"] for chunk in chunks]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, vectors=np.asarray(vectors, dtype=np.float32), ids=np.asarray(ids))
    (args.output.parent / "vectors_c_ids.json").write_text(
        json.dumps(ids, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(ids)} vectors ({embedder.dimensions} dims) to {args.output}")


if __name__ == "__main__":
    main()
