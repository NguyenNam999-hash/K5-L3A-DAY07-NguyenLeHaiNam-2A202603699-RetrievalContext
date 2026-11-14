"""Offline tests for Day 7 retrieval, agent behavior and benchmark integrity."""
import json
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent import AgentConfig, run_agent  # noqa: E402
from src.metrics import context_has_all_evidence, evidence_in_chunk, retrieval_summary  # noqa: E402
from src.retrieval import (  # noqa: E402
    BM25Retriever,
    Catalog,
    HybridRetriever,
    TfidfRetriever,
    infer_section,
    normalize_text,
    pack_context,
)

CHUNKS = [
    {
        "id": "doc-p001-b000",
        "text": "Alice found a tiny golden key on the glass table.",
        "source_id": "doc",
        "source": "book.txt",
        "page": 1,
        "metadata": {"section": "Chapter I - Down the Rabbit-Hole"},
    },
    {
        "id": "doc-p001-b001",
        "text": "The golden key opened a little door behind the curtain.",
        "source_id": "doc",
        "source": "book.txt",
        "page": 1,
        "metadata": {"section": "Chapter I - Down the Rabbit-Hole"},
    },
    {
        "id": "doc-p002-b000",
        "text": "The Queen ordered the gardeners to paint the roses red.",
        "source_id": "doc",
        "source": "book.txt",
        "page": 2,
        "metadata": {"section": "Chapter II - The Garden"},
    },
]


class RetrieverTests(unittest.TestCase):
    def test_bm25_prefers_exact_lexical_match(self):
        retriever = BM25Retriever(CHUNKS)
        rows = retriever.search("golden key", top_k=2)
        self.assertEqual(rows[0]["id"], "doc-p001-b000")
        self.assertEqual(rows[0]["rank"], 1)
        self.assertGreater(rows[0]["score"], 0)

    def test_tfidf_and_section_filter(self):
        retriever = TfidfRetriever(CHUNKS)
        rows = retriever.search("roses red", top_k=3, filters={"section": "Chapter II - The Garden"})
        self.assertEqual([row["id"] for row in rows], ["doc-p002-b000"])

    def test_infer_section(self):
        sections = sorted({chunk["metadata"]["section"] for chunk in CHUNKS})
        self.assertEqual(
            infer_section("In Chapter I - Down the Rabbit-Hole, what did Alice find?", sections),
            "Chapter I - Down the Rabbit-Hole",
        )
        self.assertEqual(
            infer_section("According to Chapter II, what colour were the roses?", sections),
            "Chapter II - The Garden",
        )
        self.assertIsNone(infer_section("What did Alice find?", sections))

    def test_pack_context_is_budgeted_and_deduplicated(self):
        results = [
            {**CHUNKS[0], "score": 1.0},
            {**CHUNKS[0], "score": 0.9},
            {**CHUNKS[2], "score": 0.8},
        ]
        context, ids = pack_context(results, budget=120)
        self.assertEqual(ids, ["doc-p001-b000"])
        self.assertLessEqual(len(context), 120)
        empty, no_ids = pack_context(results, budget=10)
        self.assertEqual((empty, no_ids), ("", []))

    def test_catalog_neighbors(self):
        catalog = Catalog(CHUNKS)
        neighbors = catalog.neighbors("doc-p001-b000")
        self.assertEqual({row["id"] for row in neighbors}, {"doc-p001-b000", "doc-p001-b001"})
        self.assertEqual(len(catalog.sections()), 2)

    def test_hybrid_rrf_fuses_dense_and_lexical(self):
        # Build a minimal fake with the same result shape instead of relying on
        # a real embedding model in the offline suite.
        class Dense:
            name = "dense"
            budget = 500

            def __init__(self):
                self.catalog = Catalog(CHUNKS)

            def search(self, query, *, top_k=3, filters=None):
                rows = []
                for rank, chunk in enumerate([CHUNKS[2], CHUNKS[0]], start=1):
                    rows.append(
                        {
                            "id": chunk["id"],
                            "text": chunk["text"],
                            "source_id": chunk["source_id"],
                            "source": chunk["source"],
                            "page": chunk["page"],
                            "metadata": chunk["metadata"],
                            "section": chunk["metadata"]["section"],
                            "score": 1.0 / rank,
                            "rank": rank,
                            "method": "dense",
                        }
                    )
                return rows[:top_k]

        hybrid = HybridRetriever(Dense(), BM25Retriever(CHUNKS), rrf_k=60, candidate_k=3)
        rows = hybrid.search("golden key", top_k=3)
        self.assertEqual(len(rows), 3)
        self.assertTrue(rows[0]["score"] >= rows[-1]["score"])
        self.assertTrue(all("dense_rank" in row or "bm25_rank" in row for row in rows))


class MetricsTests(unittest.TestCase):
    def test_evidence_and_context(self):
        gold = [{"source_id": "doc", "page": 1, "evidence": "tiny golden key"}]
        self.assertTrue(evidence_in_chunk(CHUNKS[0], gold[0]))
        self.assertFalse(evidence_in_chunk(CHUNKS[1], gold[0]))
        self.assertTrue(context_has_all_evidence([CHUNKS[0]], gold))
        self.assertFalse(context_has_all_evidence([CHUNKS[1]], gold))

    def test_retrieval_summary(self):
        traces = [
            {"hit1": True, "hit3": True, "context_hit": True, "first_hit_rank": 1,
             "elapsed_ms": 2.0, "context_chars": 50, "list_cost_usd": 0.0, "llm_calls": 0},
            {"hit1": False, "hit3": True, "context_hit": False, "first_hit_rank": 2,
             "elapsed_ms": 4.0, "context_chars": 60, "list_cost_usd": 0.0, "llm_calls": 0},
        ]
        summary = retrieval_summary(traces)
        self.assertEqual(summary["hit1"], 1)
        self.assertEqual(summary["hit3"], 2)
        self.assertEqual(summary["mrr"], 0.75)
        self.assertEqual(summary["context_hit"], 1)


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.total_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "list_cost_usd": 0.0,
        }

    def chat(self, messages, **kwargs):
        self.calls += 1
        if not self.responses:
            raise AssertionError("unexpected extra LLM call")
        message = self.responses.pop(0)
        self.total_usage["total_tokens"] += 10
        return {
            "message": message,
            "usage": {"total_tokens": 10, "list_cost_usd": 0.0},
            "finish_reason": "stop",
        }


class FakeRetriever:
    name = "fake"

    def __init__(self, rows=None, error=None):
        self.rows = rows or [
            {
                **CHUNKS[0],
                "score": 1.0,
                "rank": 1,
                "method": "fake",
                "section": CHUNKS[0]["metadata"]["section"],
            }
        ]
        self.catalog = Catalog(self.rows)
        self.error = error
        self.budget = 500

    def search(self, query, *, top_k=3, filters=None):
        if self.error:
            raise self.error
        return self.rows[:top_k]


def tool_call(name, arguments, call_id="call_1"):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    }


class AgentTests(unittest.TestCase):
    def config(self, **overrides):
        values = {
            "name": "test",
            "max_searches": 2,
            "max_steps": 6,
            "default_top_k": 1,
            "budget": 500,
            "enable_neighbors": False,
            "enable_filter": False,
        }
        values.update(overrides)
        return AgentConfig(**values)

    def test_search_then_answer_with_valid_citation(self):
        llm = FakeLLM(
            [
                tool_call("search", {"query": "golden key"}),
                tool_call(
                    "finalize",
                    {
                        "answer": "Alice found a tiny golden key [doc-p001-b000].",
                        "citations": ["doc-p001-b000"],
                        "insufficient": False,
                    },
                    "call_2",
                ),
            ]
        )
        result = run_agent("What did Alice find?", llm=llm, retriever=FakeRetriever(), config=self.config())
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["citations"], ["doc-p001-b000"])
        self.assertEqual(result["events"][0]["event"], "search")

    def test_invalid_citation_is_repaired_once(self):
        llm = FakeLLM(
            [
                tool_call("search", {"query": "golden key"}),
                tool_call(
                    "finalize",
                    {"answer": "Key [not-in-context]", "citations": ["not-in-context"], "insufficient": False},
                    "call_2",
                ),
                tool_call(
                    "finalize",
                    {"answer": "Key [doc-p001-b000]", "citations": ["doc-p001-b000"], "insufficient": False},
                    "call_3",
                ),
            ]
        )
        result = run_agent("What did Alice find?", llm=llm, retriever=FakeRetriever(), config=self.config())
        self.assertEqual(result["status"], "answered")
        self.assertEqual(llm.calls, 3)

    def test_insufficient_evidence_abstains(self):
        llm = FakeLLM(
            [
                tool_call("search", {"query": "golden key"}),
                tool_call(
                    "finalize",
                    {"answer": "No evidence.", "citations": [], "insufficient": True},
                    "call_2",
                ),
            ]
        )
        result = run_agent("What did Alice find?", llm=llm, retriever=FakeRetriever(), config=self.config())
        self.assertEqual(result["status"], "abstain")
        self.assertEqual(result["citations"], [])

    def test_refusal_language_is_abstention(self):
        llm = FakeLLM(
            [
                tool_call("search", {"query": "golden key"}),
                tool_call(
                    "finalize",
                    {
                        "answer": "The text never states the colour [doc-p001-b000].",
                        "citations": ["doc-p001-b000"],
                        "insufficient": False,
                    },
                    "call_2",
                ),
            ]
        )
        result = run_agent("What colour was the key?", llm=llm, retriever=FakeRetriever(), config=self.config())
        self.assertEqual(result["status"], "abstain")
        self.assertEqual(result["stop_reason"], "refusal_language")

    def test_search_limit_is_enforced(self):
        llm = FakeLLM(
            [
                tool_call("search", {"query": "one"}),
                tool_call("search", {"query": "two"}, "call_2"),
                tool_call("search", {"query": "three"}, "call_3"),
            ]
        )
        result = run_agent(
            "What did Alice find?",
            llm=llm,
            retriever=FakeRetriever(),
            config=self.config(max_searches=1),
        )
        self.assertEqual(result["status"], "abstain")
        self.assertEqual(result["searches"], 1)
        self.assertEqual(result["stop_reason"], "search_limit")

    def test_retriever_error_propagates(self):
        llm = FakeLLM([tool_call("search", {"query": "one"})])
        with self.assertRaises(ConnectionError):
            run_agent(
                "What did Alice find?",
                llm=llm,
                retriever=FakeRetriever(error=ConnectionError("db down")),
                config=self.config(),
            )

    def test_text_response_with_citation_is_accepted(self):
        llm = FakeLLM(
            [
                tool_call("search", {"query": "golden key"}),
                {"role": "assistant", "content": "A tiny golden key [doc-p001-b000]."},
            ]
        )
        result = run_agent("What did Alice find?", llm=llm, retriever=FakeRetriever(), config=self.config())
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["citations"], ["doc-p001-b000"])


class BenchmarkIntegrityTests(unittest.TestCase):
    def test_counts_evidence_and_unanswerable_checks(self):
        chunks = json.loads((ROOT / "data" / "day6" / "chunks_c.json").read_text(encoding="utf-8"))
        questions = json.loads((ROOT / "benchmark" / "questions.json").read_text(encoding="utf-8"))
        by_page = {}
        for chunk in chunks:
            by_page.setdefault((chunk["source_id"], chunk["page"]), []).append(chunk)
        self.assertEqual(len(questions), 24)
        self.assertEqual(sum(1 for q in questions if q["split"] == "dev" and q.get("answerable", True)), 15)
        self.assertEqual(sum(1 for q in questions if q["split"] == "test" and q.get("answerable", True)), 5)
        self.assertEqual(sum(1 for q in questions if not q.get("answerable", True)), 4)
        corpus = "\n".join(chunk["text"] for chunk in chunks)
        for question in questions:
            if not question.get("answerable", True):
                for pattern in question.get("absence_checks", []):
                    self.assertIsNone(re.search(pattern, corpus, flags=re.IGNORECASE), question["id"])
                continue
            for gold in question["gold"]:
                candidates = by_page[(gold["source_id"], gold["page"])]
                evidence = normalize_text(gold["evidence"])
                self.assertTrue(
                    any(evidence in normalize_text(chunk["text"]) for chunk in candidates),
                    f"{question['id']} evidence missing",
                )


if __name__ == "__main__":
    unittest.main()
