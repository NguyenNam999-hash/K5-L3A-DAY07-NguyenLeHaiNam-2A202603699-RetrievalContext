# Day 6 data export

`chunks_c.json` là export của **chiến lược C** Day 6 (paragraph ≤1.000 ký tự + metadata
`section`), gồm 180 chunk. Mỗi bản ghi dùng schema Day 7:

```json
{"id": "...", "text": "...", "source_id": "alice-1865", "source": "...", "page": 1, "metadata": {"section": "..."}}
```

`source_manifest.json` giữ nguồn, license, hash, số trang và mô tả chunking từ Day 6.
Nguồn là public domain (Project Gutenberg #11).

Tạo lại (cần Day 6 pgvector đang chạy):

```bash
.venv/bin/python scripts/export_day6.py
```

Vector snapshot (`vectors_c.npz`) không commit; tạo lại bằng:

```bash
.venv/bin/python scripts/build_snapshot.py
.venv/bin/python scripts/check_store_parity.py --limit 5
```
