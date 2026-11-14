"""Verify the advanced benchmark counts, groups, evidence and absence checks."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    chunks = json.loads((ROOT / "advanced" / "index" / "chunks.json").read_text(encoding="utf-8"))
    questions = json.loads((ROOT / "advanced" / "benchmark" / "questions.json").read_text(encoding="utf-8"))
    by_page_type = {}
    for chunk in chunks:
        by_page_type.setdefault((chunk["page"], chunk["content_type"]), []).append(chunk)
    corpus = "\n".join(chunk["text"] for chunk in chunks)
    ids = [q["id"] for q in questions]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate IDs")
    counts = {
        split: {
            "answerable": sum(1 for q in questions if q["split"] == split and q.get("answerable", True)),
            "unanswerable": sum(1 for q in questions if q["split"] == split and not q.get("answerable", True)),
        }
        for split in ("dev", "test")
    }
    groups = {
        split: {
            group: sum(1 for q in questions if q["split"] == split and q["group"] == group)
            for group in ("table", "image", "chart", "multi_hop", "text", "unanswerable")
        }
        for split in ("dev", "test")
    }
    if counts["dev"]["answerable"] < 15 or counts["dev"]["unanswerable"] < 3:
        raise SystemExit(f"dev counts too low: {counts['dev']}")
    if counts["test"]["answerable"] < 24 or counts["test"]["unanswerable"] < 6:
        raise SystemExit(f"test counts too low: {counts['test']}")
    minimum = {"table": 6, "image": 5, "chart": 5, "multi_hop": 4, "text": 4}
    for group, required in minimum.items():
        if groups["test"][group] < required:
            raise SystemExit(f"test group {group} has {groups['test'][group]}, need {required}")
    for question in questions:
        if not question.get("answerable", True):
            for pattern in question.get("absence_checks", []):
                if re.search(pattern, corpus, flags=re.IGNORECASE):
                    raise SystemExit(f"{question['id']}: absence pattern matched: {pattern}")
            continue
        for item in question["gold"]:
            candidates = by_page_type.get((int(item["page"]), item["content_type"]), [])
            evidence = " ".join(item["evidence"].split()).casefold()
            if not any(evidence in " ".join(chunk["text"].split()).casefold() for chunk in candidates):
                raise SystemExit(f"{question['id']}: evidence not found: {item['evidence'][:80]!r}")
    print(json.dumps({"questions": len(questions), "counts": counts, "groups": groups}, indent=2))
    print("Advanced benchmark OK")


if __name__ == "__main__":
    main()
