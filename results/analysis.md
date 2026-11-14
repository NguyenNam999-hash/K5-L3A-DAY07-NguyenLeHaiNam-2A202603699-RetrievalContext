# Day 7 — Phân tích kết quả (base lab)

Toàn bộ số liệu lấy từ các lần chạy thật trong `results/`. Cấu hình được chốt trên
dev và ghi tại `results/selection.json` trước khi chạy test.

## 1. Dữ liệu và cấu hình

| Thành phần | Giá trị |
|---|---|
| Corpus | *Alice's Adventures in Wonderland*, Project Gutenberg #11, public domain |
| Đơn vị nội dung | 39 trang/đoạn, 142.777 ký tự |
| Chunk | Chiến lược C của Day 6: paragraph ≤1.000 ký tự + metadata `section`, 180 chunk |
| Embedding | `BAAI/bge-base-en-v1.5`, 768 chiều, FastEmbed local |
| Vector store | PostgreSQL 16 + pgvector, bảng `chunks`, cosine exact scan |
| Agent model | `deepseek-v4.1-flash` qua OpenAI-compatible endpoint |
| Benchmark | 24 câu: dev 15 có đáp án + 2 thiếu evidence; test 5 có đáp án + 2 thiếu evidence |

Benchmark test được tạo mới, không dùng lại `q16–q20` của Day 6. Hash test tại thời
điểm chốt: `335bb122ff7d0967b9ca1cab0c6fed08399027e8870f38544497c8a92e61d9d3`.

## 2. Retrieval trên dev

| Cấu hình | n | Hit@1 | Evidence Hit@3 | Context Evidence Rate | MRR | Median ms | List cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF (baseline) | 15 | 12/15 (80,0%) | 12/15 (80,0%) | 3/15 (20,0%) | 0,834 | 0,2 | $0 |
| Dense | 15 | 9/15 (60,0%) | 12/15 (80,0%) | 12/15 (80,0%) | 0,700 | 27,6 | $0 |
| BM25 | 15 | 13/15 (86,7%) | 13/15 (86,7%) | 13/15 (86,7%) | 0,897 | 0,9 | $0 |
| Hybrid RRF | 15 | 12/15 (80,0%) | 13/15 (86,7%) | 15/15 (100,0%) | 0,863 | 28,5 | $0 |
| Dense + section filter | 15 | 9/15 (60,0%) | 13/15 (86,7%) | 13/15 (86,7%) | 0,750 | 40,5 | $0 |
| Hybrid + section filter | 15 | 13/15 (86,7%) | 14/15 (93,3%) | 15/15 (100,0%) | 0,902 | 29,9 | $0 |
| **Hybrid + filter + rerank** | 15 | **15/15 (100,0%)** | **15/15 (100,0%)** | **15/15 (100,0%)** | **1,000** | 7.253,5 | **$0,00508** |

## 3. Retrieval trên test

| Cấu hình | n | Hit@1 | Evidence Hit@3 | Context Evidence Rate | MRR | Median ms | List cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF | 5 | 3/5 (60,0%) | 5/5 (100,0%) | 0/5 (0,0%) | 0,800 | 0,2 | $0 |
| Dense | 5 | 2/5 (40,0%) | 4/5 (80,0%) | 3/5 (60,0%) | 0,567 | 29,0 | $0 |
| BM25 | 5 | 5/5 (100,0%) | 5/5 (100,0%) | 5/5 (100,0%) | 1,000 | 0,9 | $0 |
| Hybrid | 5 | 4/5 (80,0%) | 4/5 (80,0%) | 5/5 (100,0%) | 0,850 | 31,4 | $0 |
| Dense + filter | 5 | 2/5 (40,0%) | 4/5 (80,0%) | 3/5 (60,0%) | 0,567 | 31,3 | $0 |
| Hybrid + filter | 5 | 4/5 (80,0%) | 4/5 (80,0%) | 5/5 (100,0%) | 0,850 | 27,8 | $0 |
| **Hybrid + filter + rerank** | 5 | **5/5 (100,0%)** | **5/5 (100,0%)** | **5/5 (100,0%)** | **1,000** | 5.315,0 | **$0,00107** |

## 4. Agent trên dev

| Cấu hình | Tool use | Số lượt gọi / câu | Grounded 0/1/2; TB | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm | Context TB /2 |
|---|---|---:|---|---:|---:|---:|---:|---:|
| Baseline (dense, top-3, 2.400 ký tự) | 17/17 | 1,353 | 2/0/13; **1,733** | 14/14 | 13/13 | 2/2 | 2/15 | chưa chấm |
| Hybrid + filter (top-5, 4.800 ký tự) | 17/17 | 1,706 | 0/0/15; **2,000** | 15/15 | 15/15 | 2/2 | 0/15 | chưa chấm |
| **Selected (rerank, top-6, 6.000 ký tự)** | 17/17 | 2,118 | 0/0/15; **2,000** | 17/17 | 15/15 | 2/2 | 0/15 | chưa chấm |

## 5. Agent trên test

| Cấu hình | Tool use | Số lượt gọi / câu | Grounded 0/1/2; TB | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm | Context 0/1/2; TB |
|---|---|---:|---|---:|---:|---:|---:|---|
| Baseline | 7/7 | 1,286 | 0/0/5; **2,000** | 6/6 | 5/5 | 2/2 | 0/5 | 0/1/4; **1,800** |
| **Selected** | 7/7 | 2,000 | 0/0/5; **2,000** | 8/8 | 6/6 | **1/2** | 0/5 | 0/0/5; **2,000** |

## 6. Phân tích lỗi

### Lỗi dev 1 — baseline q08 (Pig and Pepper)

- Câu hỏi: “what does Alice think it would be murder to do?”
- Baseline dense top-3 không đưa chunk `alice-1865-p019-b000` (chứa “wouldn’t it be murder to leave it behind?”) vào context; agent hết lượt search và abstain.
- Nguyên nhân: query diễn đạt lại (“murder”) không trùng từ với đoạn evidence; dense ưu tiên các chunk nói về Pig and Pepper nhưng không chứa câu trả lời.
- Sau khi đổi sang hybrid + filter: BM25 khớp “murder/leave it behind”, section filter giữ Chapter VI; `p019-b000` vào top-1, agent trả lời đúng, grounded 2.
- Bài học: lexical fusion bổ sung đúng điểm yếu của dense; filter giảm distractor nhưng không thay thế ranking.

### Lỗi dev 2 — baseline q11 (Cheshire Cat)

- Câu hỏi: Alice đề xuất gì để giải quyết tranh chấp về Cheshire Cat?
- Baseline dense không lấy được `alice-1865-p027-b002` trong hai lượt search, chạm giới hạn và abstain.
- Nguyên nhân: query nhắc “disagreement/Cheshire Cat” trong khi evidence dùng “appealed to her to settle the question” và “It belongs to the Duchess”; độ tương đồng embedding thấp.
- Sau khi đổi sang hybrid + filter: BM25 khớp “Duchess”, chunk `p027-b002` lên top-1; agent trả lời đúng, grounded 2.
- Bài học: câu hỏi diễn đạt lại cần lexical + dense; tăng số lượt search không bù được nếu top-k ban đầu sai.

### Lỗi test — selected u03 (unanswerable, ambiguous)

- Câu hỏi: “What name does the Duchess give her baby?”
- Selected trả lời: “The Duchess calls her baby ‘Pig!’” và cite `p018-b000`.
- Benchmark khai báo câu này thiếu evidence vì baby không có tên riêng; “Pig!” là câu Duchess quát vào baby, không phải tên. Tuy nhiên cách hỏi “what name” khiến “Pig!” trở thành một cách đọc hợp lý.
- Tôi giữ nguyên test và chấm đây là **false abstention** (abstention 1/2) thay vì sửa test sau khi đã xem kết quả. Đây là lỗi chất lượng benchmark, không phải lỗi retrieval; câu nên được viết lại thành “What is the baby’s proper name?” ở một bộ test mới.
- Đây là lý do selected thấp hơn baseline ở abstention test (1/2 so với 2/2); báo cáo giữ nguyên kết quả.

## 7. Chọn cấu hình và đánh đổi

Tiêu chí công bố trước khi chạy test (`results/selection.json`):

1. Evidence Hit@3 trên dev.
2. Context Evidence Rate.
3. Grounded answer mean.
4. Latency và list cost.

Theo tiêu chí 1–3, **hybrid + section filter + LLM rerank** thắng dev (15/15 Hit@3,
15/15 context, grounded 2,00). Tuy nhiên test cho thấy:

- Answer grounded của baseline và selected đều 2,00/2 trên 5 câu test.
- Selected có context tốt hơn (2,00 so với 1,80) nhưng abstention kém hơn do u03 mơ hồ.
- Selected dùng nhiều token hơn (136.250 so với 41.716) và lâu hơn (27,3 s so với 13,3 s mỗi câu).
- Rerank retrieval có list cost $0,00107 cho 5 câu test; hybrid+filter gần như miễn phí.

Kết luận trung thực: rerank cải thiện rõ trên dev nhưng **không xác nhận cải thiện
answer quality trên test**; nó đổi chi phí lấy Hit@1/Context. Nếu ưu tiên chi phí,
hybrid + filter là lựa chọn thực dụng hơn; nếu ưu tiên ranking tuyệt đối, rerank là
lựa chọn đã chốt. Báo cáo giữ cả hai và không che kết quả test kém hơn.

## 8. Giới hạn

- Một tài liệu, một ngôn ngữ, 24 câu; test chỉ 5 câu có đáp án nên một câu làm lệch 20 điểm phần trăm.
- Nhãn grounded/citation/abstention do AI soạn nháp từ trace; sinh viên cần đọc lại trước khi coi là nhãn người.
- u03 mơ hồ như đã nêu; không sửa test sau khi đã xem kết quả.
- Rerank dùng cùng model với agent, có thể không độc lập; prompt và model version ảnh hưởng kết quả.
- Snapshot vector được kiểm tra parity top-10 trên 5 câu dev; không phải bảo đảm cho mọi query.
- Chưa đo độ ổn định nhiều lần chạy; temperature 0 nhưng endpoint vẫn có thể dao động.

## 9. Tái lập

```bash
# Day 7 environment
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Day 6 database must be running on 127.0.0.1:55433
.venv/bin/python scripts/check_setup.py
.venv/bin/python scripts/export_day6.py
.venv/bin/python benchmark/verify_benchmark.py

# Retrieval comparison
.venv/bin/python -m src.evaluate_retrieval --split dev
.venv/bin/python -m src.evaluate_retrieval --split test

# Agents
.venv/bin/python -m src.run_agent_benchmark --split dev --config baseline
.venv/bin/python -m src.run_agent_benchmark --split dev --config hybrid_filter
.venv/bin/python -m src.run_agent_benchmark --split dev --config selected
.venv/bin/python -m src.run_agent_benchmark --split test --config baseline
.venv/bin/python -m src.run_agent_benchmark --split test --config selected

# Labels and metrics
.venv/bin/python scripts/build_reviews.py --config baseline --split dev
.venv/bin/python scripts/build_reviews.py --config hybrid_filter --split dev
.venv/bin/python scripts/build_reviews.py --config selected --split dev
.venv/bin/python scripts/build_test_reviews.py --config baseline --split test
.venv/bin/python scripts/build_test_reviews.py --config selected --split test

# Offline tests
.venv/bin/python -m unittest discover -s tests -v
```
