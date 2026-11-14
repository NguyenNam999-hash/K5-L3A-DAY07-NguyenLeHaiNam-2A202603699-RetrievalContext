"""Read-only Day 6 pgvector adapter plus a verified local snapshot mirror."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Protocol

_IDENTIFIER_RE = re.compile(r"^[a-z_][a-z0-9_]*$")
_ALLOWED_FILTERS = {
    "document_id",
    "source",
    "page",
    "section",
    "category",
    "year",
    "language",
    "document_type",
    "audience",
    "version",
}


class VectorStore(Protocol):
    dimensions: int

    def search(
        self,
        query_vector: list[float],
        *,
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
        strategy: str | None = None,
    ) -> list[dict[str, Any]]: ...

    def all_chunks(self, *, strategy: str | None = None) -> list[dict[str, Any]]: ...


def _vector_literal(vector: list[float]) -> str:
    if not vector:
        raise ValueError("empty vector")
    return "[" + ",".join(f"{float(value):.8f}" for value in vector) + "]"


def _row_to_chunk(row: tuple[Any, ...]) -> dict[str, Any]:
    return {
        "id": row[0],
        "source_id": row[1],
        "source": row[2],
        "page": int(row[3]),
        "text": row[4],
        "metadata": row[5] if isinstance(row[5], dict) else json.loads(row[5] or "{}"),
        "score": float(row[6]) if len(row) > 6 and row[6] is not None else None,
    }


class PgVectorStore:
    """Read-only access to the Day 6 ``chunks`` table."""

    def __init__(self, database_url: str, dimensions: int, table: str = "chunks") -> None:
        if not database_url:
            raise ValueError("database_url is required")
        if not _IDENTIFIER_RE.match(table):
            raise ValueError(f"invalid table name: {table}")
        if not isinstance(dimensions, int) or dimensions <= 0:
            raise ValueError("dimensions must be a positive integer")
        self.database_url = database_url
        self.dimensions = dimensions
        self.table = table

    def _connect(self):
        import psycopg

        return psycopg.connect(self.database_url, connect_timeout=10)

    def search(
        self,
        query_vector: list[float],
        *,
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
        strategy: str | None = None,
    ) -> list[dict[str, Any]]:
        if len(query_vector) != self.dimensions:
            raise ValueError(
                f"Query dimension mismatch: expected {self.dimensions}, got {len(query_vector)}"
            )
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        clauses: list[str] = []
        args: list[Any] = []
        if strategy:
            clauses.append("strategy = %s")
            args.append(strategy)
        filter_json = None
        if filters:
            cleaned: dict[str, Any] = {}
            for key, value in filters.items():
                if not _IDENTIFIER_RE.match(key) or key not in _ALLOWED_FILTERS:
                    raise ValueError(f"Filter key is not allowed: {key}")
                if value is not None:
                    cleaned[key] = value
            if cleaned:
                clauses.append("metadata @> %s::jsonb")
                filter_json = json.dumps(cleaned, ensure_ascii=False)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"""
            SELECT chunk_id, document_id, source, page, text, metadata,
                   1 - (embedding <=> %s::vector) AS score
            FROM {self.table}
            {where}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        vector_literal = _vector_literal(query_vector)
        sql_args: list[Any] = [vector_literal, *args]
        if filter_json is not None:
            sql_args.append(filter_json)
        sql_args.extend([vector_literal, top_k])
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(sql, sql_args)
                rows = cursor.fetchall()
        return [_row_to_chunk(row) for row in rows]

    def all_chunks(self, *, strategy: str | None = None) -> list[dict[str, Any]]:
        where = "WHERE strategy = %s" if strategy else ""
        args = (strategy,) if strategy else ()
        sql = f"""
            SELECT chunk_id, document_id, source, page, text, metadata, NULL
            FROM {self.table} {where} ORDER BY chunk_id
        """
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(sql, args)
                rows = cursor.fetchall()
        return [_row_to_chunk(row) for row in rows]

    def count(self, *, strategy: str | None = None) -> int:
        where = "WHERE strategy = %s" if strategy else ""
        args = (strategy,) if strategy else ()
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(f"SELECT COUNT(*) FROM {self.table} {where}", args)
                return int(cursor.fetchone()[0])


class SnapshotStore:
    """Exact-cosine mirror of the Day 6 index, built from real embeddings.

    The snapshot is opt-in (``--store snapshot``) and must pass the parity check
    in ``scripts/check_store_parity.py`` before it is used for reported results.
    """

    def __init__(self, chunks_path: Path, vectors_path: Path, ids_path: Path | None = None) -> None:
        import numpy as np

        self.chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        data = np.load(vectors_path)
        if "vectors" in data:
            self.vectors = np.asarray(data["vectors"], dtype=np.float32)
            self.ids = [str(value) for value in data["ids"]] if "ids" in data else [c["id"] for c in self.chunks]
        else:
            raise ValueError("vectors file must contain 'vectors'")
        if ids_path and ids_path.exists():
            self.ids = json.loads(ids_path.read_text(encoding="utf-8"))
        if len(self.ids) != len(self.chunks) or len(self.vectors) != len(self.chunks):
            raise ValueError("snapshot chunks, ids and vectors have different lengths")
        self.index = {chunk_id: i for i, chunk_id in enumerate(self.ids)}
        self.dimensions = int(self.vectors.shape[1])
        if self.dimensions != 768:
            raise ValueError(f"unexpected snapshot dimension: {self.dimensions}")
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        self.vectors = self.vectors / np.clip(norms, 1e-12, None)

    def search(
        self,
        query_vector: list[float],
        *,
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
        strategy: str | None = None,
    ) -> list[dict[str, Any]]:
        import numpy as np

        if len(query_vector) != self.dimensions:
            raise ValueError(
                f"Query dimension mismatch: expected {self.dimensions}, got {len(query_vector)}"
            )
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query = np.asarray(query_vector, dtype=np.float32)
        query = query / max(float(np.linalg.norm(query)), 1e-12)
        scores = self.vectors @ query
        order = np.argsort(-scores)
        results: list[dict[str, Any]] = []
        for index in order:
            chunk = self.chunks[int(index)]
            if filters and any(str(chunk.get("metadata", {}).get(key)) != str(value) for key, value in filters.items() if value is not None):
                continue
            results.append(
                {
                    "id": chunk["id"],
                    "source_id": chunk["source_id"],
                    "source": chunk["source"],
                    "page": int(chunk["page"]),
                    "text": chunk["text"],
                    "metadata": chunk.get("metadata", {}),
                    "score": float(scores[int(index)]),
                }
            )
            if len(results) >= top_k:
                break
        return results

    def all_chunks(self, *, strategy: str | None = None) -> list[dict[str, Any]]:
        return [
            {
                "id": chunk["id"],
                "source_id": chunk["source_id"],
                "source": chunk["source"],
                "page": int(chunk["page"]),
                "text": chunk["text"],
                "metadata": chunk.get("metadata", {}),
                "score": None,
            }
            for chunk in self.chunks
        ]
