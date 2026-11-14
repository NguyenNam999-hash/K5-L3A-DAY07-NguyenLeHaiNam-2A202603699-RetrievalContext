# Advanced benchmark

`questions.json` has 48 questions built by `advanced/scripts/build_benchmark.py`:

| Split | Answerable | Unanswerable | Total |
|---|---:|---:|---:|
| dev | 15 | 3 | 18 |
| test | 24 | 6 | 30 |

Test groups:

| Group | Required | Actual |
|---|---:|---:|
| Table | 6 | 6 |
| Image/map/diagram | 5 | 5 |
| Chart | 5 | 5 |
| Multi-hop | 4 | 4 |
| Text | 4 | 4 |
| Missing evidence | 6 | 6 |

Each answerable item has:

```json
{
  "id": "at-t01",
  "split": "test",
  "group": "table",
  "question": "...",
  "expected_answer": "...",
  "answerable": true,
  "gold": [
    {"page": 26, "content_type": "table", "evidence": "| World | 6.3 | ... |",
     "table_id": "p026-t01", "row": "World", "column": "2024f"}
  ]
}
```

`page` is the PDF page index (1–220), not the printed page number. Evidence must be a
verbatim substring of a chunk with that page and content type. For multi-hop, `gold` has
at least two entries from different pages.

Unanswerable items have `answerable=false`, `expected_behavior=search_and_abstain`,
`absence_reason`, and `absence_checks` regexes that must not match the corpus.

Pass/fail follows the advanced rule: an answerable item passes only when the answer is
correct and complete (including units/conditions), every citation is in the context, and
the cited page/region supports the whole answer. An unanswerable item passes only when the
agent refuses or clearly states insufficient evidence without unsupported claims.

Verify:

```bash
.venv/bin/python advanced/scripts/verify_benchmark.py
```
