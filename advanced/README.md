# Advanced challenge — 220-page multimodal report

## Source

**World Bank. 2024. *Global Economic Prospects, June 2024*.** Washington, DC: World Bank.
doi:10.1596/978-1-4648-2058-8. **CC BY 3.0 IGO**.

- Landing page: <https://openknowledge.worldbank.org/entities/publication/b5f5a90f-3229-482f-b8c5-d3c4e9ecbad5>
- PDF URL and SHA-256: `advanced/source/manifest.json`
- The PDF is **not committed**. Run `advanced/scripts/fetch_document.py`; it verifies the SHA-256.

## Pipeline

1. **Fetch + manifest** — `advanced/scripts/fetch_document.py`.
2. **Extract** — `advanced/scripts/extract_document.py`:
   - 220 pages, full text layer, page headings.
   - 31 pages with tables; table cells read from the table bbox and clipped text
     (this preserves decimals such as `6.3`; `table.extract()` produced `63 .`).
   - 72 pages with charts/maps; charts rendered and described by the vision model.
   - Six chapter-opener locator maps use manually verified descriptions because the
     vision model classified the small locator maps as decorative.
   - Text chunks ≤1.200 characters; table chunks keep rows/columns/units; figure chunks
     keep page, caption, bbox and description.
3. **Index** — `advanced/scripts/build_index.py`:
   - `BAAI/bge-base-en-v1.5`, 768 dims.
   - PostgreSQL + pgvector table **`advanced_chunks`**, cosine exact search.
   - 359 chunks: 249 text, 38 table, 65 chart, 7 image.
4. **Agent** — `advanced/scripts/run_agent.py`:
   - Tools: `search` (with `content_type` filter), `get_page`, `finalize`.
   - 14.000-character context budget; up to 6 searches, 14 steps.
   - Citation IDs must be in context and in the answer; otherwise abstain.
5. **Benchmark** — `advanced/benchmark/questions.json`:
   - dev: 15 answerable + 3 unanswerable.
   - test: 24 answerable (6 table, 5 image, 5 chart, 4 multi-hop, 4 text) + 6 unanswerable.
   - Evidence is a verbatim substring of a page/type chunk; `verify_benchmark.py` checks it.
  `page` means the **PDF page index** (1–220), not the printed page number (the printed
  number is usually 22 lower).

## Results

### Dev

| Pipeline | Answerable answered | Unanswerable abstained | Errors | Tokens | Cost |
|---|---:|---:|---:|---:|---:|
| Dense baseline | 12/15 | 3/3 | 0 | 235.434 | $0,0260 |
| **Selected (hybrid + rerank)** | **15/15** | **3/3** | 0 | 1.179.891 | $0,1394 |

### Test (selected, frozen before test)

| Group | Passed / total | Rate |
|---|---:|---:|
| Table | 6/6 | 100% |
| Image/map | 5/5 | 100% |
| Chart | 5/5 | 100% |
| Multi-hop | 3/4 | 75% |
| Text | 3/4 | 75% |
| Unanswerable | 6/6 | 100% |
| **All** | **28/30** | **93,3%** |

The 90% threshold is **met**. Two answerable questions (`at-x01`, `at-m04`) were refused
because the final answer did not produce a valid citation; the evidence was present in
the retrieved context. They are kept as failures and analyzed in `results/analysis.md`.

- Test traces: `advanced/results/agent/selected/test/traces.json`
- Test labels: `advanced/results/agent/selected/test/reviews.json`
- Test metrics: `advanced/results/agent/selected/test/metrics.json`
- Selection freeze: `advanced/results/selection.json`
- Extraction summary: `advanced/results/extraction_summary.json` (copy of the generated
  `advanced/index/extract_summary.json`; the index itself is rebuildable and not committed)

## Commands

```bash
# 1. Source
.venv/bin/python advanced/scripts/fetch_document.py

# 2. Extract + vision (first run calls the vision endpoint; cache is resumable)
.venv/bin/python advanced/scripts/extract_document.py

# 3. Index (needs Day 6 pgvector running)
.venv/bin/python advanced/scripts/build_index.py

# 4. Verify + run
.venv/bin/python advanced/scripts/verify_benchmark.py
.venv/bin/python advanced/scripts/run_agent.py --split dev --config selected
.venv/bin/python advanced/scripts/run_agent.py --split dev --config baseline
.venv/bin/python advanced/scripts/freeze_selection.py
.venv/bin/python advanced/scripts/run_agent.py --split test --config selected
.venv/bin/python advanced/scripts/build_reviews.py --config selected --split test
```

## Extraction coverage and limits

- 220 pages processed; 194 pages have text; 26 pages are cover/TOC/blank or have no
  indexable text. Those are listed in `advanced/index/pages.jsonl` with zero character counts.
- 38 tables extracted; 72 chart/map figures described (65 chart, 7 image). 66 vision calls
  were cached in `advanced/cache/` (not committed); the first extraction cost about $0.054.
- Tables with merged cells or multi-line headers can still be misread; the six test tables
  were checked manually against the rendered pages.
- Chart values are read from the rendered figure and vision description; the five test chart
  questions were checked against the page image. Small labels may be missed.
- The six locator maps are manually described and verified; the vision model called two of
  them decorative.
- The source is born-digital, so no OCR was needed. A scanned PDF would need an OCR stage.
- This is one document; results do not generalize to other reports or layouts.
