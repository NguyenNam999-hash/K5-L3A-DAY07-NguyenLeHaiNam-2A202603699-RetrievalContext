# Day 7 — Agentic RAG

**Bài cá nhân · Python 3.11+ · tiếp nối Day 6**

Dùng embedding và vector store từ Day 6 [K5-L3-Day06-NguyenLeHaiNam-DataIngestionEngineering](../K5-L3-Day06-NguyenLeHaiNam-DataIngestionEngineering) để xây một agent biết tìm bằng chứng, trả lời có nguồn và nhận biết khi không đủ thông tin. Bạn tự chọn kiến trúc, số tool và cách tổ chức retrieval.

> **Tùy chọn:** [Gợi ý dùng AI để lập trình](AGENT_SETUP.md) giúp bạn làm lab; phần này không bắt buộc và không tính điểm.

## Bài làm — tóm tắt

Solution nằm ở `src/`, dùng **Day 6 chiến lược C** (180 chunk, pgvector,
`BAAI/bge-base-en-v1.5` 768 chiều) và agent `deepseek-v4.1-flash`.

- **Benchmark:** 24 câu (`benchmark/questions.json`): dev 15+2, test 5+2; test mới, không
  dùng lại `q16–q20` Day 6.
- **Retrieval:** TF-IDF, dense pgvector, BM25, hybrid RRF, dense/hybrid + section filter,
  hybrid + filter + LLM rerank. Dev chốt **hybrid + filter + rerank** (15/15 Hit@3);
  test 5/5 Hit@1, 5/5 Hit@3, 5/5 context.
- **Agent:** tool `search`, `get_neighbors`, `list_sections`, `finalize`; citation phải
  nằm trong context và xuất hiện trong answer; thiếu evidence thì abstain. Trace đầy đủ
  trong `results/agent/`.
- **Kết quả:** retrieval dev/test, agent dev/test, nhãn nháp và phân tích lỗi ở
  `results/analysis.md`; báo cáo ở
  `report/individual/2A202603699_NguyenLeHaiNam.md`.
- **Advanced:** World Bank *Global Economic Prospects, June 2024* (220 trang, CC BY 3.0
  IGO), 359 chunk text/table/chart/image, benchmark 30 test (24 answerable + 6 thiếu
  evidence). Test đạt **28/30 = 93,3%** (ngưỡng 90%); trace/nhãn/phân tích ở
  `advanced/`. PDF không commit; tải bằng `advanced/scripts/fetch_document.py`.
- **Lưu ý:** nhãn grounded/citation/abstention do AI soạn nháp từ trace; sinh viên cần đọc
  lại. Test đã chốt hash trong `results/selection.json` trước khi chạy.

Chạy nhanh:

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/check_setup.py
.venv/bin/python scripts/export_day6.py
.venv/bin/python benchmark/verify_benchmark.py
.venv/bin/python -m src.evaluate_retrieval --split dev
.venv/bin/python -m src.run_agent_benchmark --split dev --config selected
.venv/bin/python -m unittest discover -s tests -v

# Advanced (220-page report; PDF is downloaded and verified, not committed)
.venv/bin/python advanced/scripts/fetch_document.py
.venv/bin/python advanced/scripts/extract_document.py
.venv/bin/python advanced/scripts/build_index.py
.venv/bin/python advanced/scripts/verify_benchmark.py
.venv/bin/python advanced/scripts/run_agent.py --split dev --config selected
.venv/bin/python advanced/scripts/run_agent.py --split test --config selected
.venv/bin/python advanced/scripts/build_reviews.py --config selected --split test
```

## Bài cần làm

![Agent tìm tài liệu trong kho Day 6 và trả lời có nguồn](assets/retrieval-pipeline.svg)

1. Kết nối dữ liệu, embedding và vector store Day 6.
2. Xây agent dùng model thật và ít nhất một khả năng truy xuất dữ liệu. Agent có thể dùng nhiều tool và nhiều lượt gọi.
3. Thử **ít nhất 3 phương pháp retrieval khác nhau**, trong đó có dense retrieval dùng vector store Day 6. Tự chọn các phương pháp còn lại; xem [gợi ý](GUIDE.md#2-thử-các-phương-pháp-retrieval).
4. Thử nhiều cấu hình, benchmark, phân tích lỗi và chọn cấu hình tốt trên **dev**; sau đó đánh giá trên **test**.

Benchmark tối thiểu có **24 câu**: dev gồm 15 câu có đáp án và 2 câu thiếu bằng chứng; test gồm 5 câu có đáp án và 2 câu thiếu bằng chứng. Xem [benchmark](BENCHMARK.md) và [rubric](RUBRIC.md).

## Thử thách nâng cao · +15 điểm

Index một tài liệu thực tế từ **100 trang** có bảng, ảnh và biểu đồ; xây agent truy xuất và trả lời các câu hỏi về nội dung đó. Hoàn thành [thử thách nâng cao](ADVANCED.md) và đạt **ít nhất 90% benchmark test** để nhận 15 điểm bonus. Phần cơ bản tối đa 85 điểm; toàn bài tối đa 100 điểm.

## Bắt đầu

1. Đọc [hướng dẫn](GUIDE.md), [rubric](RUBRIC.md) và [quy tắc](RULES.md).
2. Tạo repo cá nhân theo [cách nộp bài](SUBMISSION.md), chuẩn bị đầu ra Day 6 và model hỗ trợ tool calling.
3. Dùng hoặc thay thế [starter](starter/README.md); bạn cần kết nối model và vector store thật.
4. Chạy `python -m unittest discover -s tests -v`. Test mock chỉ kiểm tra code, không phải benchmark agent thật.

![Các bước thử nghiệm trên dev và đánh giá trên test](assets/experiment-loop.svg)

Nộp code, cách chạy, cấu hình, benchmark, trace và [báo cáo](report/individual/MSSV_HoTen.md). Thời gian ở [checkpoints](CHECKPOINTS.md) là gợi ý, có thể điều chỉnh theo lớp và phạm vi thử nghiệm.
