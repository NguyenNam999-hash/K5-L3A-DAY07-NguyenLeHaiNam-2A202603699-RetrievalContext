"""Build the 24-question Day 7 benchmark from the Day 6 q01-q15 dev set and new held-out items."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DAY6 = ROOT.parent / "K5-L3-Day06-NguyenLeHaiNam-DataIngestionEngineering"


def main() -> None:
    raw = json.loads((DAY6 / "benchmark" / "benchmark.json").read_text(encoding="utf-8"))
    dev = []
    for question in raw:
        if question["split"] != "dev":
            continue
        item = {
            "id": question["id"],
            "split": "dev",
            "type": question.get("type", "direct"),
            "question": question["question"],
            "expected_answer": question["expected_answer"],
            "gold": [
                {
                    "source_id": question["gold_source_id"],
                    "page": question["gold_page"],
                    "evidence": question["evidence_quote"],
                }
            ],
            "answerable": True,
        }
        if question.get("filter"):
            item["filter"] = question["filter"]
        dev.append(item)

    test = [
        {
            "id": "t01",
            "split": "test",
            "type": "direct",
            "question": "In the pool of tears, what does Alice say is the capital of Paris while reciting her geography lesson?",
            "expected_answer": "London.",
            "gold": [{"source_id": "alice-1865", "page": 5,
                      "evidence": "London is the capital of Paris, and Paris is the capital of Rome"}],
            "answerable": True,
        },
        {
            "id": "t02",
            "split": "test",
            "type": "paraphrase",
            "question": "According to the creatures outside the house, who is supposed to go down the chimney?",
            "expected_answer": "Bill.",
            "gold": [{"source_id": "alice-1865", "page": 12,
                      "evidence": "Who’s to go down the chimney?—Nay, _I_ shan’t! _You_ do it!—_That_ I won’t, then!—Bill’s to go down"}],
            "answerable": True,
        },
        {
            "id": "t03",
            "split": "test",
            "type": "direct",
            "question": "In Chapter VII - A Mad Tea-Party, what were the names of the three little sisters in the Dormouse's story?",
            "expected_answer": "Elsie, Lacie, and Tillie.",
            "gold": [{"source_id": "alice-1865", "page": 23,
                      "evidence": "their names were Elsie, Lacie, and Tillie"}],
            "filter": {"section": "Chapter VII - A Mad Tea-Party"},
            "answerable": True,
        },
        {
            "id": "t04",
            "split": "test",
            "type": "paraphrase",
            "question": "Where is the Mock Turtle sitting when Alice and the Gryphon first see him?",
            "expected_answer": "On a little ledge of rock.",
            "gold": [{"source_id": "alice-1865", "page": 30,
                      "evidence": "sitting sad and lonely on a little ledge of rock"}],
            "answerable": True,
        },
        {
            "id": "t05",
            "split": "test",
            "type": "direct",
            "question": "How do the officers of the court suppress the guinea-pig that cheers?",
            "expected_answer": "They put it head first into a canvas bag tied at the mouth and sit on it.",
            "gold": [{"source_id": "alice-1865", "page": 36,
                      "evidence": "They had a large canvas bag, which tied up at the mouth with strings: into this they slipped the guinea-pig, head first, and then sat upon it."}],
            "answerable": True,
        },
    ]

    unanswerable = [
        {
            "id": "u01", "split": "dev", "question": "How old is Alice?",
            "answerable": False, "expected_behavior": "search_and_abstain",
            "absence_reason": "The narrative never states Alice's age.",
            "absence_checks": [r"Alice.{0,40}(is|was)\s+(seven|eight|nine|ten|eleven|twelve)\s+years\s+old", r"Alice'?s\s+age"],
        },
        {
            "id": "u02", "split": "dev", "question": "What is the Queen of Hearts' first name?",
            "answerable": False, "expected_behavior": "search_and_abstain",
            "absence_reason": "Only the title 'Queen of Hearts' is used; no given name appears.",
            "absence_checks": [r"Queen\s+of\s+Hearts.{0,30}(named|called)", r"Queen[^.]{0,30}(first|given)\s+name"],
        },
        {
            "id": "u03", "split": "test", "question": "What name does the Duchess give her baby?",
            "answerable": False, "expected_behavior": "search_and_abstain",
            "absence_reason": "The baby is never named; it is described as a baby and later becomes a pig.",
            "absence_checks": [r"baby.{0,30}(named|called)", r"name[^.]{0,30}baby"],
        },
        {
            "id": "u04", "split": "test", "question": "How many years did the Mock Turtle spend at school?",
            "answerable": False, "expected_behavior": "search_and_abstain",
            "absence_reason": "The Mock Turtle describes his school and subjects but never states a duration in years.",
            "absence_checks": [r"(school|education).{0,60}\d+\s+years", r"\d+\s+years.{0,60}(school|education)"],
        },
    ]

    questions = dev + test + unanswerable
    order = {"dev": 0, "test": 1}
    questions.sort(key=lambda item: (order[item["split"]], item["id"]))
    output = ROOT / "benchmark" / "questions.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(questions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(questions)} questions to {output}")


if __name__ == "__main__":
    main()
