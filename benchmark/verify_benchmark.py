"""Validate the Day 7 benchmark against the exported Day 6 chunks."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.retrieval import infer_section, normalize_text  # noqa: E402


def main() -> None:
    chunks = json.loads((ROOT / "data" / "day6" / "chunks_c.json").read_text(encoding="utf-8"))
    questions = json.loads((ROOT / "benchmark" / "questions.json").read_text(encoding="utf-8"))
    by_source_page: dict[tuple[str, int], list[dict]] = {}
    for chunk in chunks:
        by_source_page.setdefault((chunk["source_id"], int(chunk["page"])), []).append(chunk)
    if len({chunk["id"] for chunk in chunks}) != len(chunks):
        raise SystemExit("duplicate chunk IDs")
    sections = sorted({chunk["metadata"].get("section", "") for chunk in chunks if chunk["metadata"].get("section")})
    corpus = "\n".join(chunk["text"] for chunk in chunks)

    ids = [item["id"] for item in questions]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate question IDs")
    counts = {
        split: {
            "answerable": sum(1 for q in questions if q["split"] == split and q.get("answerable", True)),
            "unanswerable": sum(1 for q in questions if q["split"] == split and not q.get("answerable", True)),
        }
        for split in ("dev", "test")
    }
    if counts["dev"]["answerable"] < 15 or counts["test"]["answerable"] < 5:
        raise SystemExit(f"not enough answerable questions: {counts}")
    if counts["dev"]["unanswerable"] < 2 or counts["test"]["unanswerable"] < 2:
        raise SystemExit(f"not enough unanswerable questions: {counts}")

    for question in questions:
        qid = question["id"]
        if question["split"] not in ("dev", "test"):
            raise SystemExit(f"{qid}: invalid split")
        if not isinstance(question.get("question"), str) or not question["question"].strip():
            raise SystemExit(f"{qid}: missing question")
        if not question.get("answerable", True):
            if question.get("expected_behavior") != "search_and_abstain":
                raise SystemExit(f"{qid}: missing expected_behavior")
            if not question.get("absence_reason"):
                raise SystemExit(f"{qid}: missing absence_reason")
            for pattern in question.get("absence_checks", []):
                if re.search(pattern, corpus, flags=re.IGNORECASE):
                    raise SystemExit(f"{qid}: absence check unexpectedly matched: {pattern}")
            continue
        expected = question.get("expected_answer")
        if not isinstance(expected, str) or not expected.strip():
            raise SystemExit(f"{qid}: missing expected_answer")
        gold = question.get("gold")
        if not isinstance(gold, list) or not gold:
            raise SystemExit(f"{qid}: missing gold")
        for item in gold:
            if not all(key in item for key in ("source_id", "page", "evidence")):
                raise SystemExit(f"{qid}: incomplete gold item")
            candidates = by_source_page.get((item["source_id"], int(item["page"])), [])
            evidence = normalize_text(item["evidence"])
            if not evidence:
                raise SystemExit(f"{qid}: empty evidence")
            if not any(evidence in normalize_text(chunk["text"]) for chunk in candidates):
                raise SystemExit(f"{qid}: evidence not found in source/page chunk")
        filter_value = (question.get("filter") or {}).get("section")
        if filter_value:
            inferred = infer_section(question["question"], sections)
            if inferred != filter_value:
                raise SystemExit(
                    f"{qid}: filter {filter_value!r} is not inferred from the question (got {inferred!r})"
                )
            gold_page = int(gold[0]["page"])
            page_chunks = by_source_page.get((gold[0]["source_id"], gold_page), [])
            if not any(chunk["metadata"].get("section") == filter_value for chunk in page_chunks):
                raise SystemExit(f"{qid}: filter does not match the gold page metadata")

    print("Benchmark OK")
    print(json.dumps({"questions": len(questions), "splits": counts}, indent=2))


if __name__ == "__main__":
    main()
