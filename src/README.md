# Code — Day 7

Thư mục này chứa solution thật (không phải starter).

| File | Công dụng |
|---|---|
| `config.py` | Đọc `.env`, đường dẫn Day 6, model/store |
| `embeddings.py` | FastEmbed `BAAI/bge-base-en-v1.5`, 768 chiều |
| `store.py` | pgvector read-only + snapshot cosine exact |
| `retrieval.py` | Dense, BM25, TF-IDF, hybrid RRF, LLM rerank, context packing, filter inference |
| `llm.py` | OpenAI-compatible client, retry, usage/cost |
| `agent.py` | Agent tool-calling, citation validation, abstention, trace |
| `evaluate_retrieval.py` | So sánh retrieval trên dev/test |
| `run_agent_benchmark.py` | Chạy agent thật trên một split |
| `metrics.py` | Metric retrieval/agent/context |

Chạy từ thư mục gốc:

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/check_setup.py
.venv/bin/python scripts/export_day6.py
.venv/bin/python benchmark/verify_benchmark.py

.venv/bin/python -m src.evaluate_retrieval --split dev
.venv/bin/python -m src.run_agent_benchmark --split dev --config selected
.venv/bin/python -m unittest discover -s tests -v
```

`starter/` vẫn giữ nguyên làm tài liệu tham chiếu; solution nằm ở `src/`.
