"""Offline integrity tests for the advanced benchmark."""
import json
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.retrieval import normalize_text  # noqa: E402


class AdvancedBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        chunks_path = ROOT / "advanced" / "index" / "chunks.json"
        if not chunks_path.exists():
            raise unittest.SkipTest(
                "advanced index not built; run advanced/scripts/fetch_document.py and extract_document.py"
            )
        cls.chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        cls.questions = json.loads(
            (ROOT / "advanced" / "benchmark" / "questions.json").read_text(encoding="utf-8")
        )
        cls.by_page_type = {}
        for chunk in cls.chunks:
            cls.by_page_type.setdefault((chunk["page"], chunk["content_type"]), []).append(chunk)
        cls.corpus = "\n".join(chunk["text"] for chunk in cls.chunks)

    def test_counts_and_groups(self):
        counts = {
            split: {
                "answerable": sum(
                    1 for q in self.questions if q["split"] == split and q.get("answerable", True)
                ),
                "unanswerable": sum(
                    1 for q in self.questions if q["split"] == split and not q.get("answerable", True)
                ),
            }
            for split in ("dev", "test")
        }
        self.assertEqual(counts["dev"], {"answerable": 15, "unanswerable": 3})
        self.assertEqual(counts["test"], {"answerable": 24, "unanswerable": 6})
        test_groups = {
            group: sum(1 for q in self.questions if q["split"] == "test" and q["group"] == group)
            for group in ("table", "image", "chart", "multi_hop", "text", "unanswerable")
        }
        self.assertEqual(test_groups["table"], 6)
        self.assertEqual(test_groups["image"], 5)
        self.assertEqual(test_groups["chart"], 5)
        self.assertEqual(test_groups["multi_hop"], 4)
        self.assertEqual(test_groups["text"], 4)
        self.assertEqual(test_groups["unanswerable"], 6)

    def test_evidence_and_absence(self):
        for question in self.questions:
            if not question.get("answerable", True):
                for pattern in question.get("absence_checks", []):
                    self.assertIsNone(re.search(pattern, self.corpus, flags=re.IGNORECASE), question["id"])
                continue
            for item in question["gold"]:
                candidates = self.by_page_type.get((item["page"], item["content_type"]), [])
                evidence = normalize_text(item["evidence"])
                self.assertTrue(
                    any(evidence in normalize_text(chunk["text"]) for chunk in candidates),
                    f"{question['id']} evidence missing",
                )

    def test_table_cells_preserve_decimals(self):
        table_chunks = [chunk for chunk in self.chunks if chunk["content_type"] == "table"]
        world = next(chunk for chunk in table_chunks if chunk["id"] == "adv-p026-t01")
        self.assertIn("| World | 6.3 | 3.0 | 2.6 | 2.6 | 2.7 | 2.7 |", world["text"])


if __name__ == "__main__":
    unittest.main()
