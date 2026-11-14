"""pgvector storage for the advanced 220-page index."""
from __future__ import annotations

import json
import re
from typing import Any

_IDENTIFIER_RE = re.compile(r"^[a-z_][a-z0-9_]*$")
_TABLE = "advanced_chunks"


def _vector_literal(vector: list[float]) -> str:
    if not vector:
        raise ValueError("empty vector")
    return "[" + ",".join(f"{float(value):.8f}" for value in vector) + "]"


class AdvancedPgStore:
    def __init__(self, database_url: str, dimensions: int) -> None:
        if not database_url:
            raise ValueError("database_url is required")
        if not isinstance(dimensions, int) or dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.database_url = database_url
        self.dimensions = dimensions

    def _connect(self):
        import psycopg

        return psycopg.connect(self.database_url, connect_timeout=10)

    def create_schema(self) -> None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cursor.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {_TABLE} (
                        chunk_id TEXT PRIMARY KEY,
                        page INTEGER NOT NULL,
                        content_type TEXT NOT NULL,
                        section TEXT NOT NULL DEFAULT '',
                        text TEXT NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        embedding vector({self.dimensions}) NOT NULL
                    )
                    """
                )
                cursor.execute(f"CREATE INDEX IF NOT EXISTS {_TABLE}_page_idx ON {_TABLE} (page)")
                cursor.execute(
                    f"CREATE INDEX IF NOT EXISTS {_TABLE}_type_idx ON {_TABLE} (content_type)"
                )

    def upsert(self, chunks: list[dict[str, Any]], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        for chunk, vector in zip(chunks, vectors):
            if len(vector) != self.dimensions:
                raise ValueError(f"dimension mismatch for {chunk['id']}")
        payload = []
        for chunk, vector in zip(chunks, vectors):
            metadata = dict(chunk.get("metadata") or {})
            metadata["page"] = chunk["page"]
            metadata["content_type"] = chunk["content_type"]
            payload.append(
                (
                    chunk["id"],
                    int(chunk["page"]),
                    chunk["content_type"],
                    chunk.get("section", ""),
                    chunk["text"],
                    json.dumps(metadata, ensure_ascii=False),
                    _vector_literal(vector),
                )
            )
        ids = [chunk["id"] for chunk in chunks]
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"DELETE FROM {_TABLE} WHERE NOT (chunk_id = ANY(%s))", (ids,)
                )
                cursor.executemany(
                    f"""
                    INSERT INTO {_TABLE}
                        (chunk_id, page, content_type, section, text, metadata, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::vector)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        page = EXCLUDED.page,
                        content_type = EXCLUDED.content_type,
                        section = EXCLUDED.section,
                        text = EXCLUDED.text,
                        metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding
                    """,
                    payload,
                )

    def search(
        self,
        query_vector: list[float],
        *,
        top_k: int = 6,
        filters: dict[str, Any] | None = None,
        strategy: str | None = None,
    ) -> list[dict[str, Any]]:
        if len(query_vector) != self.dimensions:
            raise ValueError(
                f"query dimension mismatch: expected {self.dimensions}, got {len(query_vector)}"
            )
        clauses = []
        args: list[Any] = []
        if filters:
            for key, value in filters.items():
                if value is None:
                    continue
                if key == "page":
                    clauses.append("page = %s")
                    args.append(int(value))
                elif key == "content_type":
                    clauses.append("content_type = %s")
                    args.append(str(value))
                else:
                    raise ValueError(f"filter not allowed: {key}")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"""
            SELECT chunk_id, page, content_type, section, text, metadata,
                   1 - (embedding <=> %s::vector) AS score
            FROM {_TABLE} {where}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        vector_literal = _vector_literal(query_vector)
        sql_args: list[Any] = [vector_literal, *args, vector_literal, top_k]
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(sql, sql_args)
                rows = cursor.fetchall()
        return [
            {
                "id": row[0],
                "page": int(row[1]),
                "content_type": row[2],
                "section": row[3],
                "text": row[4],
                "metadata": row[5] if isinstance(row[5], dict) else json.loads(row[5] or "{}"),
                "score": float(row[6]),
            }
            for row in rows
        ]

    def all_chunks(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT chunk_id, page, content_type, section, text, metadata FROM {_TABLE} ORDER BY chunk_id"
                )
                rows = cursor.fetchall()
        return [
            {
                "id": row[0],
                "page": int(row[1]),
                "content_type": row[2],
                "section": row[3],
                "text": row[4],
                "metadata": row[5] if isinstance(row[5], dict) else json.loads(row[5] or "{}"),
                "score": None,
            }
            for row in rows
        ]

    def page_chunks(self, page: int) -> list[dict[str, Any]]:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT chunk_id, page, content_type, section, text, metadata FROM {_TABLE} WHERE page = %s ORDER BY content_type, chunk_id",
                    (page,),
                )
                rows = cursor.fetchall()
        return [
            {
                "id": row[0],
                "page": int(row[1]),
                "content_type": row[2],
                "section": row[3],
                "text": row[4],
                "metadata": row[5] if isinstance(row[5], dict) else json.loads(row[5] or "{}"),
                "score": None,
            }
            for row in rows
        ]

    def count(self) -> int:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(f"SELECT COUNT(*) FROM {_TABLE}")
                return int(cursor.fetchone()[0])
