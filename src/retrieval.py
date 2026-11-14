"""Retrieval methods: dense pgvector, BM25, hybrid RRF, rerank, and context packing."""
from __future__ import annotations

import json
import math
import re
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Protocol

from .embeddings import Embedder
from .store import VectorStore

_WS_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’_-]*")
_CHUNK_RE = re.compile(r"-p(\d+)-[a-z](\d+)$")
_QUOTE_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
    }
)
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "did", "do",
    "does", "for", "from", "had", "has", "have", "he", "her", "his", "how", "i",
    "in", "into", "is", "it", "its", "of", "on", "or", "she", "that", "the",
    "their", "them", "there", "they", "this", "to", "was", "were", "what", "when",
    "where", "which", "who", "why", "will", "with", "would", "you", "your",
}


def normalize_text(text: str | None) -> str:
    text = unicodedata.normalize("NFC", text or "").translate(_QUOTE_TRANSLATION)
    return _WS_RE.sub(" ", text).strip().casefold()


def tokenize(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFC", text or "").translate(_QUOTE_TRANSLATION)
    tokens = []
    for token in _WORD_RE.findall(normalized.casefold()):
        token = token.strip("'’_-")
        if token and token not in _STOPWORDS and len(token) > 1:
            tokens.append(token)
    return tokens


def as_chunk(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata") or {}
    return {
        "id": row["id"],
        "text": row.get("text", ""),
        "source_id": row.get("source_id"),
        "source": row.get("source"),
        "page": int(row.get("page")) if row.get("page") is not None else None,
        "metadata": metadata,
        "section": metadata.get("section", ""),
        "score": row.get("score"),
    }


def _result(chunk: dict[str, Any], *, rank: int, score: float, method: str, **extra: Any) -> dict[str, Any]:
    return {
        "id": chunk["id"],
        "text": chunk.get("text", ""),
        "source_id": chunk.get("source_id"),
        "source": chunk.get("source"),
        "page": chunk.get("page"),
        "metadata": chunk.get("metadata", {}),
        "section": (chunk.get("metadata") or {}).get("section", chunk.get("section", "")),
        "score": float(score),
        "rank": rank,
        "method": method,
        **extra,
    }


class Catalog:
    """In-memory chunk catalog used for BM25, neighbors and id lookups."""

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.chunks = [as_chunk(chunk) for chunk in chunks]
        self.by_id = {chunk["id"]: chunk for chunk in self.chunks}
        self.by_page: dict[int, list[dict[str, Any]]] = {}
        for chunk in self.chunks:
            self.by_page.setdefault(int(chunk["page"]), []).append(chunk)
        for page_chunks in self.by_page.values():
            page_chunks.sort(key=lambda c: c["id"])

    def get(self, chunk_id: str) -> dict[str, Any] | None:
        return self.by_id.get(chunk_id)

    def neighbors(self, chunk_id: str, radius: int = 1) -> list[dict[str, Any]]:
        chunk = self.by_id.get(chunk_id)
        if not chunk:
            return []
        match = _CHUNK_RE.search(chunk_id)
        if not match:
            return []
        page = int(match.group(1))
        index = int(match.group(2))
        page_chunks = self.by_page.get(page, [])
        result = []
        for candidate in page_chunks:
            candidate_match = _CHUNK_RE.search(candidate["id"])
            if candidate_match and abs(int(candidate_match.group(2)) - index) <= radius:
                result.append(candidate)
        return result

    def sections(self) -> list[str]:
        return sorted({chunk["section"] for chunk in self.chunks if chunk.get("section")})


def infer_section(question: str, sections: list[str]) -> str | None:
    """Return a section explicitly named in the question, or None.

    This is query-derived only. It never reads gold labels.
    """
    question_norm = normalize_text(question)
    best: str | None = None
    for section in sections:
        section_norm = normalize_text(section)
        if section_norm and section_norm in question_norm:
            if best is None or len(section_norm) > len(normalize_text(best)):
                best = section
    if best:
        return best
    # Accept "Chapter IV" even when the chapter subtitle is omitted.
    chapter_match = re.search(r"\bchapter\s+([ivxlcdm]+)\b", question_norm)
    if chapter_match:
        prefix = f"chapter {chapter_match.group(1)}"
        matches = [section for section in sections if normalize_text(section).startswith(prefix)]
        if len(matches) == 1:
            return matches[0]
    return None


def pack_context(results: list[dict[str, Any]], budget: int = 3000) -> tuple[str, list[str]]:
    """Deduplicate and pack whole chunks under a character budget."""
    if budget < 1:
        raise ValueError("budget must be positive")
    blocks: list[str] = []
    ids: list[str] = []
    seen: set[str] = set()
    for chunk in results:
        chunk_id = chunk.get("id")
        if not chunk_id or chunk_id in seen:
            continue
        seen.add(chunk_id)
        section = (chunk.get("metadata") or {}).get("section", chunk.get("section", ""))
        header = f"[{chunk_id} | {chunk.get('source', '')} | page {chunk.get('page')}"
        if section:
            header += f" | {section}"
        header += "]"
        block = f"{header}\n{chunk.get('text', '')}"
        if len("\n\n".join(blocks + [block])) <= budget:
            blocks.append(block)
            ids.append(chunk_id)
    return "\n\n".join(blocks), ids


class Retriever(Protocol):
    name: str
    budget: int
    catalog: Catalog

    def search(
        self,
        query: str,
        *,
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]: ...


class DenseRetriever:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        *,
        strategy: str,
        chunks: list[dict[str, Any]],
        name: str = "dense",
        budget: int = 3000,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.strategy = strategy
        self.name = name
        self.budget = budget
        self.catalog = Catalog(chunks)

    def search(
        self,
        query: str,
        *,
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        query_vector = self.embedder.embed_query(query)
        rows = self.store.search(
            query_vector, top_k=top_k, filters=filters, strategy=self.strategy
        )
        return [
            _result(as_chunk(row), rank=index, score=row.get("score") or 0.0, method=self.name)
            for index, row in enumerate(rows, start=1)
        ]


class BM25Retriever:
    def __init__(
        self,
        chunks: list[dict[str, Any]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        name: str = "bm25",
        budget: int = 3000,
    ) -> None:
        self.catalog = Catalog(chunks)
        self.name = name
        self.budget = budget
        self.k1 = k1
        self.b = b
        self.docs = [chunk for chunk in self.catalog.chunks if chunk.get("text")]
        self.doc_tokens = [tokenize(chunk["text"]) for chunk in self.docs]
        self.doc_len = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl = sum(self.doc_len) / max(len(self.doc_len), 1)
        df = Counter()
        for tokens in self.doc_tokens:
            df.update(set(tokens))
        n = len(self.docs)
        self.idf = {
            term: math.log(1 + (n - count + 0.5) / (count + 0.5))
            for term, count in df.items()
        }

    def search(
        self,
        query: str,
        *,
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        query_terms = tokenize(query)
        if not query_terms:
            return []
        scores: list[tuple[float, int]] = []
        for index, chunk in enumerate(self.docs):
            if filters:
                metadata = chunk.get("metadata") or {}
                if any(
                    str(metadata.get(key)) != str(value)
                    for key, value in filters.items()
                    if value is not None
                ):
                    continue
            tokens = self.doc_tokens[index]
            if not tokens:
                continue
            counts = Counter(tokens)
            length = self.doc_len[index]
            score = 0.0
            for term in set(query_terms):
                tf = counts.get(term, 0)
                if not tf:
                    continue
                idf = self.idf.get(term, 0.0)
                denominator = tf + self.k1 * (1 - self.b + self.b * length / max(self.avgdl, 1e-9))
                score += idf * tf * (self.k1 + 1) / denominator
            if score > 0:
                scores.append((score, index))
        scores.sort(key=lambda pair: (-pair[0], self.docs[pair[1]]["id"]))
        return [
            _result(self.docs[index], rank=rank, score=score, method=self.name)
            for rank, (score, index) in enumerate(scores[:top_k], start=1)
        ]


class TfidfRetriever:
    """Offline lexical baseline, independent of the starter implementation."""

    def __init__(self, chunks: list[dict[str, Any]], *, name: str = "tfidf", budget: int = 800) -> None:
        self.catalog = Catalog(chunks)
        self.name = name
        self.budget = budget
        self.docs = [chunk for chunk in self.catalog.chunks if chunk.get("text")]
        self.doc_tokens = [tokenize(chunk["text"]) for chunk in self.docs]
        df = Counter()
        for tokens in self.doc_tokens:
            df.update(set(tokens))
        n = len(self.docs)
        self.idf = {term: math.log((1 + n) / (1 + count)) + 1 for term, count in df.items()}
        self.vectors = [self._vector(Counter(tokens)) for tokens in self.doc_tokens]

    def _vector(self, counts: Counter[str]) -> dict[str, float]:
        vector = {
            term: count * self.idf[term]
            for term, count in counts.items()
            if term in self.idf
        }
        norm = math.sqrt(sum(value * value for value in vector.values())) or 1.0
        return {term: value / norm for term, value in vector.items()}

    def search(
        self,
        query: str,
        *,
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        query_vector = self._vector(Counter(tokenize(query)))
        scores: list[tuple[float, int]] = []
        for index, vector in enumerate(self.vectors):
            chunk = self.docs[index]
            if filters:
                metadata = chunk.get("metadata") or {}
                if any(
                    str(metadata.get(key)) != str(value)
                    for key, value in filters.items()
                    if value is not None
                ):
                    continue
            score = sum(value * vector.get(term, 0.0) for term, value in query_vector.items())
            if score > 0:
                scores.append((score, index))
        scores.sort(key=lambda pair: (-pair[0], self.docs[pair[1]]["id"]))
        return [
            _result(self.docs[index], rank=rank, score=score, method=self.name)
            for rank, (score, index) in enumerate(scores[:top_k], start=1)
        ]


class HybridRetriever:
    def __init__(
        self,
        dense: Retriever,
        lexical: Retriever,
        *,
        rrf_k: int = 60,
        candidate_k: int = 30,
        name: str = "hybrid",
        budget: int = 3000,
    ) -> None:
        self.dense = dense
        self.lexical = lexical
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k
        self.name = name
        self.budget = budget
        self.catalog = dense.catalog

    def search(
        self,
        query: str,
        *,
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        candidate_k = max(self.candidate_k, top_k)
        dense_rows = self.dense.search(query, top_k=candidate_k, filters=filters)
        lexical_rows = self.lexical.search(query, top_k=candidate_k, filters=filters)
        fused: dict[str, dict[str, Any]] = {}
        for rank, row in enumerate(dense_rows, start=1):
            entry = fused.setdefault(row["id"], {**row, "rrf_score": 0.0, "dense_rank": None, "bm25_rank": None})
            entry["rrf_score"] += 1.0 / (self.rrf_k + rank)
            entry["dense_rank"] = rank
            entry["dense_score"] = row["score"]
        for rank, row in enumerate(lexical_rows, start=1):
            entry = fused.setdefault(row["id"], {**row, "rrf_score": 0.0, "dense_rank": None, "bm25_rank": None})
            entry["rrf_score"] += 1.0 / (self.rrf_k + rank)
            entry["bm25_rank"] = rank
            entry["bm25_score"] = row["score"]
        ordered = sorted(fused.values(), key=lambda row: (-row["rrf_score"], row["id"]))
        return [
            _result(row, rank=rank, score=row["rrf_score"], method=self.name,
                    dense_rank=row.get("dense_rank"), bm25_rank=row.get("bm25_rank"),
                    dense_score=row.get("dense_score"), bm25_score=row.get("bm25_score"))
            for rank, row in enumerate(ordered[:top_k], start=1)
        ]


class LLMReranker:
    def __init__(self, llm: Any, *, candidate_chars: int = 1100) -> None:
        self.llm = llm
        self.candidate_chars = candidate_chars
        self.last_usage: dict[str, Any] | None = None
        self.last_error: str | None = None

    def rerank(self, question: str, candidates: list[dict[str, Any]], top_k: int) -> list[str]:
        if not candidates:
            return []
        self.last_error = None
        last_content = ""
        last_error: Exception | None = None
        for attempt in range(3):
            candidate_chars = (
                self.candidate_chars
                if attempt == 0
                else max(350, self.candidate_chars // (2 if attempt == 1 else 3))
            )
            max_tokens = 5000 if attempt < 2 else 8000
            response_format = {"type": "json_object"} if attempt < 2 else None
            lines = []
            for candidate in candidates:
                text = candidate.get("text", "")[:candidate_chars]
                lines.append(
                    f"ID: {candidate['id']} | page {candidate.get('page')} | "
                    f"section {candidate.get('section', '')}\n{text}"
                )
            prompt = (
                "Rank the candidate chunks by whether they contain the evidence needed to "
                "answer the question. Return ONLY a JSON object with key `ids`, an array of "
                "chunk IDs ordered best first, containing at most "
                f"{max(top_k, 1)} IDs. Do not answer the question. Do not include prose.\n\n"
                f"Question: {question}\n\nCandidates:\n\n" + "\n\n---\n\n".join(lines)
            )
            response = self.llm.chat(
                [
                    {"role": "system", "content": "You are a precise retrieval reranker. Output JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=max_tokens,
                response_format=response_format,
            )
            self.last_usage = response.get("usage")
            last_content = response.get("message", {}).get("content") or ""
            try:
                payload = json.loads(last_content)
                ids = payload.get("ids")
                if not isinstance(ids, list) or any(not isinstance(value, str) for value in ids):
                    raise ValueError("ids must be a list of strings")
                allowed = {candidate["id"] for candidate in candidates}
                ordered = [chunk_id for chunk_id in ids if chunk_id in allowed]
                for candidate in candidates:
                    if candidate["id"] not in ordered:
                        ordered.append(candidate["id"])
                return ordered[:top_k]
            except Exception as exc:
                last_error = exc
                continue
        # The reranker is an optional ranking stage. Degrade to the fusion
        # ordering, but make the failure visible in the trace.
        self.last_error = f"Reranker returned invalid JSON after 3 attempts: {last_content[:300]!r} ({last_error})"
        return [candidate["id"] for candidate in candidates[:top_k]]


class RerankRetriever:
    def __init__(
        self,
        base: Retriever,
        reranker: LLMReranker,
        *,
        candidate_k: int = 20,
        name: str = "hybrid_filter_rerank",
        budget: int = 6000,
    ) -> None:
        self.base = base
        self.reranker = reranker
        self.candidate_k = candidate_k
        self.name = name
        self.budget = budget
        self.catalog = base.catalog

    def search(
        self,
        query: str,
        *,
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        candidates = self.base.search(query, top_k=self.candidate_k, filters=filters)
        started = time.perf_counter()
        self.reranker.last_error = None
        ordered_ids = self.reranker.rerank(query, candidates, top_k)
        by_id = {candidate["id"]: candidate for candidate in candidates}
        results = []
        for rank, chunk_id in enumerate(ordered_ids, start=1):
            row = by_id[chunk_id]
            results.append(
                _result(row, rank=rank, score=1.0 / rank, method=self.name,
                        original_rank=row.get("rank"), rerank_usage=self.reranker.last_usage,
                        rerank_failed=bool(self.reranker.last_error),
                        rerank_error=self.reranker.last_error)
            )
        for row in results:
            row["rerank_seconds"] = round(time.perf_counter() - started, 4)
        return results


@dataclass
class RetrievalConfig:
    name: str
    kind: str
    top_k: int = 3
    budget: int = 3000
    filters: bool = False
    description: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


BASE_CONFIGS: dict[str, RetrievalConfig] = {
    "tfidf": RetrievalConfig("tfidf", "tfidf", top_k=3, budget=800,
                             description="TF-IDF cosine, top-3, no filter"),
    "dense": RetrievalConfig("dense", "dense", top_k=3, budget=2400,
                             description="BGE + pgvector cosine, top-3, no filter"),
    "bm25": RetrievalConfig("bm25", "bm25", top_k=3, budget=2400,
                            description="BM25, top-3, no filter"),
    "hybrid": RetrievalConfig("hybrid", "hybrid", top_k=3, budget=4800,
                              description="Dense + BM25 RRF, top-3, no filter"),
    "dense_filter": RetrievalConfig("dense_filter", "dense", top_k=3, budget=2400,
                                    filters=True,
                                    description="Dense + section filter inferred from explicit chapter mention"),
    "hybrid_filter": RetrievalConfig("hybrid_filter", "hybrid", top_k=5, budget=4800,
                                     filters=True,
                                     description="Hybrid RRF + section filter, top-5"),
    "hybrid_filter_rerank": RetrievalConfig(
        "hybrid_filter_rerank", "hybrid_rerank", top_k=6, budget=6000, filters=True,
        description="Hybrid RRF top-20 + LLM listwise rerank + section filter, top-6",
    ),
}


def build_retriever(
    config: RetrievalConfig,
    *,
    embedder: Embedder,
    store: VectorStore,
    strategy: str,
    chunks: list[dict[str, Any]],
    llm: Any | None = None,
) -> Retriever:
    dense = DenseRetriever(embedder, store, strategy=strategy, chunks=chunks,
                           name="dense", budget=config.budget)
    lexical = BM25Retriever(chunks, name="bm25", budget=config.budget)
    if config.kind == "tfidf":
        return TfidfRetriever(chunks, name=config.name, budget=config.budget)
    if config.kind == "dense":
        dense.name = config.name
        return dense
    if config.kind == "bm25":
        lexical.name = config.name
        return lexical
    hybrid = HybridRetriever(dense, lexical, name=config.name, budget=config.budget)
    if config.kind == "hybrid":
        return hybrid
    if config.kind == "hybrid_rerank":
        if llm is None:
            raise ValueError("LLM is required for hybrid_rerank")
        return RerankRetriever(
            hybrid,
            LLMReranker(llm),
            name=config.name,
            budget=config.budget,
        )
    raise ValueError(f"unknown retrieval kind: {config.kind}")
