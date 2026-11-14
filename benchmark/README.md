# Benchmark Day 7

`questions.json` có 24 câu, tạo bởi `scripts/build_benchmark.py`:

| Split | Có đáp án | Thiếu evidence | Tổng |
|---|---:|---:|---:|
| dev | 15 (`q01–q15`) | 2 (`u01–u02`) | 17 |
| test | 5 (`t01–t05`) | 2 (`u03–u04`) | 7 |

- `q01–q15` kế thừa dev Day 6; `t01–t05` là câu mới từ các trang Day 6 không dùng ở dev.
- `gold` gồm `source_id`, `page`, `evidence`; evidence phải nằm trọn trong một chunk
  strategy C đúng nguồn/trang.
- Câu thiếu evidence có `absence_reason` và `absence_checks` (regex phải không khớp corpus).
- Filter `section` chỉ có khi câu hỏi nêu rõ tên chương; `verify_benchmark.py` kiểm tra
  filter suy ra được từ câu hỏi và khớp metadata trang gold.
- Test không được dùng để chọn cấu hình; hash được ghi trong `results/selection.json`.

Chạy kiểm tra:

```bash
.venv/bin/python benchmark/verify_benchmark.py
```
