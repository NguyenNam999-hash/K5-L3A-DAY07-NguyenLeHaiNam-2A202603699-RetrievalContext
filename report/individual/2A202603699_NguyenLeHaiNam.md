# Báo cáo Day 7

Họ và tên: Nguyễn Lê Hải Nam · MSSV: 2A202603699 · Lớp: L3A

## Dữ liệu và benchmark

- **Nguồn, quyền sử dụng:** *Alice's Adventures in Wonderland* của Lewis Carroll,
  Project Gutenberg eBook #11 (`https://www.gutenberg.org/files/11/11-0.txt`), public domain tại
  Hoa Kỳ, dùng theo Project Gutenberg License. Snapshot và hash nằm trong
  `data/day6/source_manifest.json`; kế thừa từ Day 6.
- **Số trang/chunk:** 39 đơn vị nội dung, 142.777 ký tự; dùng **chiến lược C** của Day 6
  (paragraph ≤1.000 ký tự + metadata `section`), **180 chunk**. ID, `source`, `page`,
  `section` được giữ nguyên. Bản export Day 7: `data/day6/chunks_c.json`.
- **Vector store/embedding:** PostgreSQL 16 + pgvector, bảng `chunks`, metric cosine, exact
  scan (không ANN); `BAAI/bge-base-en-v1.5`, 768 chiều, FastEmbed/ONNX chạy local.
  Query dùng `query_embed` của cùng model. Snapshot local được kiểm tra parity top-10 với
  pgvector trên 5 câu dev (`scripts/check_store_parity.py`).
- **Agent model:** `deepseek-v4.1-flash` qua endpoint OpenAI-compatible trong `.env`
  (cần header `x-opencode-session`). Temperature 0.
- **Chia dev/test:** 24 câu trong `benchmark/questions.json`:
  - dev: `q01–q15` (15 câu có đáp án, kế thừa Day 6) + `u01–u02` (2 câu thiếu evidence).
  - test: `t01–t05` (5 câu mới, lấy từ các trang Day 6 không dùng ở dev) + `u03–u04`
    (2 câu thiếu evidence).
  - Không dùng lại `q16–q20` của Day 6 vì bộ đó đã được xem trong Day 6.
- **Cách xác định câu có/thiếu evidence:** câu có đáp án có `gold` gồm `source_id`, `page`,
  `evidence`; `benchmark/verify_benchmark.py` kiểm tra evidence là substring nguyên văn
  trong chunk đúng nguồn/trang. Câu thiếu evidence có `answerable=false`,
  `expected_behavior=search_and_abstain`, `absence_reason` và các regex `absence_checks`
  không khớp toàn corpus.
- **Tiêu chí chấm:** retrieval dùng Hit@1/Evidence Hit@3 với evidence nằm trọn trong chunk
  đúng nguồn/trang; context dùng Context Evidence Rate; agent dùng grounded 0/1/2,
  citation support, abstention và false abstention.

## Retrieval và thử nghiệm

Bảy cấu hình, cùng corpus, chunk, embedding model và bộ câu hỏi. Filter `section` chỉ được
suy ra từ câu hỏi khi câu nêu rõ tên chương (ví dụ “In Chapter VI - Pig and Pepper”), không
dùng gold. Cấu hình chốt trên dev trước khi chạy test: `results/selection.json`.

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---:|---:|---:|---|
| Baseline TF-IDF | TF-IDF cosine, top-3, không filter, 800 ký tự | Baseline lexical offline | dev | 12/15 (80,0%) | 12/15 (80,0%) | 3/15 (20,0%) | 0,2 ms; $0 |
| Dense | BGE + pgvector, top-3, không filter, 2.400 ký tự | Dense trên vector store Day 6 | dev | 9/15 (60,0%) | 12/15 (80,0%) | 12/15 (80,0%) | 27,6 ms; $0 |
| BM25 | BM25, top-3, không filter, 2.400 ký tự | Lexical mạnh hơn TF-IDF cho tên riêng/từ hiếm | dev | 13/15 (86,7%) | 13/15 (86,7%) | 13/15 (86,7%) | 0,9 ms; $0 |
| Hybrid RRF | Dense + BM25, RRF, top-3, 4.800 ký tự | Fusion bù điểm yếu semantic/lexical | dev | 12/15 (80,0%) | 13/15 (86,7%) | 15/15 (100,0%) | 28,5 ms; $0 |
| Dense + filter | Dense, top-3, filter `section`, 2.400 ký tự | Filter giảm distractor chương khác | dev | 9/15 (60,0%) | 13/15 (86,7%) | 13/15 (86,7%) | 40,5 ms; $0 |
| Hybrid + filter | Hybrid RRF, top-5, filter `section`, 4.800 ký tự | Fusion + filter + top-5 | dev | 13/15 (86,7%) | 14/15 (93,3%) | 15/15 (100,0%) | 29,9 ms; $0 |
| **Cấu hình đã chọn** | Hybrid + filter + LLM listwise rerank top-20 → top-6, 6.000 ký tự | Rerank xếp evidence lên đầu | dev | **15/15 (100,0%)** | **15/15 (100,0%)** | **15/15 (100,0%)** | 7.253,5 ms; $0,00508 |
| Baseline TF-IDF | như trên | Đánh giá test | test | 3/5 (60,0%) | 5/5 (100,0%) | 0/5 (0,0%) | 0,2 ms; $0 |
| Dense | như trên | Đánh giá test | test | 2/5 (40,0%) | 4/5 (80,0%) | 3/5 (60,0%) | 29,0 ms; $0 |
| BM25 | như trên | Đánh giá test | test | 5/5 (100,0%) | 5/5 (100,0%) | 5/5 (100,0%) | 0,9 ms; $0 |
| Hybrid + filter | như trên | Đánh giá test | test | 4/5 (80,0%) | 4/5 (80,0%) | 5/5 (100,0%) | 27,8 ms; $0 |
| **Cấu hình đã chọn** | Hybrid + filter + rerank | Đánh giá test sau khi chốt | test | **5/5 (100,0%)** | **5/5 (100,0%)** | **5/5 (100,0%)** | 5.315,0 ms; $0,00107 |

Bảng đầy đủ 7 cấu hình × 2 split nằm trong `results/retrieval/summary_dev.json` và
`summary_test.json`; trace từng câu nằm trong `results/retrieval/<config>/<split>/traces.json`.

**Hai lỗi dev và cách sửa:**

1. **q08 — Pig and Pepper:** baseline dense top-3 bỏ lỡ `alice-1865-p019-b000`
   (“wouldn’t it be murder to leave it behind?”), agent hết lượt search và abstain. Nguyên
   nhân: câu hỏi diễn đạt lại không trùng từ với evidence. Sau khi thêm BM25 + section
   filter, chunk này lên top-1 và agent trả lời đúng (grounded 2).
2. **q11 — Cheshire Cat:** baseline dense không lấy `alice-1865-p027-b002` trong hai lượt
   search rồi abstain. Evidence dùng “settle the question”/“the Duchess” trong khi query
   dùng “disagreement/Cheshire Cat”. Hybrid + filter đưa `p027-b002` lên top-1; grounded 2.

**Vì sao chọn rerank:** theo tiêu chí công bố trước (Evidence Hit@3 → Context Rate →
grounded answer → latency/cost), rerank thắng dev tuyệt đối. Tuy nhiên test cho thấy
answer grounded của baseline và selected đều 2,00/2; selected chỉ tốt hơn context
(2,00 so với 1,80) nhưng đắt hơn khoảng 3,6× token và 2× thời gian, và abstention kém hơn
do câu u03 mơ hồ. Đây là kết quả **không xác nhận cải thiện answer quality trên test**;
báo cáo giữ nguyên thay vì đổi test.

## Agent và đánh giá

- **Tool:** `search` (dense/BM25/hybrid/rerank theo cấu hình), `get_neighbors`
  (cùng trang), `list_sections`, `finalize`. Agent có thể search nhiều lượt; mỗi lượt lưu
  query, filter, top-k, kết quả, context và context IDs.
- **Điều kiện dừng:** `finalize`; hoặc giới hạn search (baseline 2, hybrid 4, selected 5);
  hoặc max steps (5/8/10). Khi hết lượt, agent phải trả lời từ context hiện có.
- **Chọn context:** dedupe theo chunk ID, ưu tiên round-robin giữa các lượt search, header
  `[id | source | page | section]`, budget ký tự. `get_neighbors` thêm chunk lân cận cùng
  trang. Baseline 2.400 ký tự; hybrid 4.800; selected 6.000.
- **Citation:** mọi citation phải nằm trong context **và** xuất hiện dạng `[chunk-id]`
  trong câu trả lời; sai thì sửa một lần, vẫn sai thì abstain. Agent cũng abstain khi
  `insufficient=true`, context rỗng, hoặc câu trả lời có ngôn ngữ từ chối rõ ràng.
- **Lỗi:** provider/store exception không bị nuốt; runner ghi `status=error` kèm traceback
  và báo riêng.

| Cấu hình / split | Tool use | Số lượt gọi | Grounded 0/1/2; TB | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm | Context 0/1/2; TB |
|---|---|---:|---|---:|---:|---:|---:|---|
| Baseline / dev | 17/17 | 1,353 | 2/0/13; 1,733 | 14/14 | 13/13 | 2/2 | 2/15 | chưa chấm |
| Hybrid+filter / dev | 17/17 | 1,706 | 0/0/15; 2,000 | 15/15 | 15/15 | 2/2 | 0/15 | chưa chấm |
| Selected / dev | 17/17 | 2,118 | 0/0/15; 2,000 | 17/17 | 15/15 | 2/2 | 0/15 | chưa chấm |
| Baseline / test | 7/7 | 1,286 | 0/0/5; 2,000 | 6/6 | 5/5 | 2/2 | 0/5 | 0/1/4; 1,800 |
| Selected / test | 7/7 | 2,000 | 0/0/5; 2,000 | 8/8 | 6/6 | **1/2** | 0/5 | 0/0/5; 2,000 |

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| t04 (selected) | Có đáp án | 1 search → `alice-1865-p030-b000` | “The Mock Turtle is sitting sad and lonely on a little ledge of rock [alice-1865-p030-b000].” | 2 — đúng, đủ, citation chứa nguyên văn evidence |
| u04 (selected) | Thiếu evidence | 2 search, không có evidence về số năm | “Insufficient evidence in the retrieved context.” | Abstention đúng — corpus không nêu số năm |
| u03 (selected) | Thiếu evidence (mơ hồ) | 2 search → `p018-b000` | “The Duchess calls her baby ‘Pig!’ …” | False abstention — “Pig!” là câu quát, không phải tên riêng; giữ nguyên để minh bạch |
| q08 (baseline) | Có đáp án | 2 search, top-3 không có `p019-b000` | “Insufficient evidence…” | 0 — bỏ lỡ evidence; hybrid+filter sửa được |
| q11 (baseline) | Có đáp án | 2 search, chạm search limit | “Insufficient evidence…” | 0 — bỏ lỡ `p027-b002`; hybrid+filter sửa được |

- **Trace/nhãn/lệnh chạy:** `results/agent/<config>/<split>/traces.json`, `cases.json`,
  `reviews.json`, `metrics.json`; tổng hợp tại `results/analysis.md`.
- **Thời gian/chi phí:** test baseline 13,3 s/câu, 41.716 token, $0,004774; selected
  27,3 s/câu, 136.250 token, $0,017394. Dev selected 47,2 s/câu do rerank lặp theo từng
  search. Đây là list cost của endpoint; gói hiện tại có thể hiển thị $0.
- **Kết quả trước/sau:** baseline dev grounded 1,733, false abstention 2/15; hybrid+filter
  và selected đều 2,000, false abstention 0/15. Trên test, baseline rẻ hơn và abstention
  tốt hơn; selected chỉ hơn context. Đánh đổi chính là chi phí/latency lấy ranking.
- **Giới hạn:** 24 câu, test 5 câu có đáp án; nhãn do AI soạn nháp, sinh viên cần duyệt;
  u03 mơ hồ; rerank dùng cùng model với agent; một tài liệu/ngôn ngữ; chưa đo nhiều lần
  chạy; kết quả gắn với snapshot và model version hiện tại.

## Thử thách nâng cao

- **Tài liệu, nguồn, quyền sử dụng:** World Bank, *Global Economic Prospects, June 2024*,
  220 trang, CC BY 3.0 IGO; manifest/hash ở `advanced/source/manifest.json`. PDF không
  commit; tải và kiểm SHA-256 bằng `advanced/scripts/fetch_document.py`.
- **Xử lý text, bảng, ảnh, biểu đồ:** 220/220 trang được xử lý; 194 trang có text layer
  (26 trang bìa/mục lục/trống). 31 trang có bảng → 38 bảng Markdown + cell rows; 72 trang
  có figure → 65 chart description bằng vision + 7 image/map (6 locator map mô tả thủ công,
  1 barcode). Bảng dùng cell bbox + clipped text để giữ dấu thập phân (`6.3`, không phải
  `63 .`). Chunk: 359 (249 text, 38 table, 65 chart, 7 image); index pgvector bảng
  `advanced_chunks`, BGE 768 chiều. Phần chưa xử lý: trang không có text, một số bảng
  merged cell và nhãn nhỏ trong chart có thể bị bỏ sót.
- **Chia benchmark và model:** dev 15 answerable + 3 unanswerable; test 24 answerable
  (6 table, 5 image, 5 chart, 4 multi-hop, 4 text) + 6 unanswerable. Test hash chốt trước
  khi chạy. Pipeline chốt trên dev: dense/hybrid + content-type filter + get_page + rerank,
  agent `deepseek-v4.1-flash`.

| Nhóm test | Đạt / tổng | Tỷ lệ | Lỗi / nhận xét |
|---|---:|---:|---|
| Bảng | 6/6 | 100% | Đúng hàng/cột/năm, giữ đơn vị |
| Ảnh, sơ đồ | 5/5 | 100% | Locator map + ISBN barcode |
| Biểu đồ | 5/5 | 100% | 38%, 2024, 14%, xu hướng 2026, 5–6% |
| Multi-hop | 3/4 | 75% | at-m04 bị từ chối do citation không hợp lệ |
| Text | 3/4 | 75% | at-x01 bị từ chối do citation không hợp lệ |
| Thiếu bằng chứng | 6/6 | 100% | Từ chối đúng, không thêm khẳng định |
| **Toàn bộ** | **28/30** | **93,3%** | **Đạt ngưỡng 90% (27/30)** |

- **Trace, nhãn, evidence:** `advanced/results/agent/selected/test/traces.json`,
  `reviews.json`, `metrics.json`; phân tích chi tiết ở `advanced/results/analysis.md`.
- **Thời gian, chi phí, lỗi:** test trung bình 63,3 giây/câu, 2.265.158 token, list cost
  $0,2534; 0 lỗi provider. Hai lỗi đều là citation/format (at-x01, at-m04), không phải
  thiếu evidence; giữ nguyên fail, không sửa test hay rerun để tăng điểm.
- **Giới hạn, tái lập:** một tài liệu/layout; 32 bảng còn lại chưa audit từng ô; 60 chart
  chưa kiểm tra từng giá trị; map mô tả thủ công; vision có thể bỏ nhãn nhỏ. Cách chạy đầy
  đủ trong `advanced/README.md`.

## Ghi công AI

Tôi dùng OpenCode (DeepSeek V4.1 Flash) để hỗ trợ viết code, chạy thí nghiệm, soạn nháp
nhãn và phân tích, gồm cả pipeline extraction/vision của phần advanced. Mọi số liệu lấy
từ trace thật trong `results/` và `advanced/results/`; không có kết quả, log hay nhãn giả.
Nhãn grounded/citation/abstention/pass-fail được AI soạn từ trace và cần sinh viên đọc lại
trước khi coi là nhãn người. Test không được dùng để chọn cấu hình; cấu hình và hash test
đã chốt trong `results/selection.json` và `advanced/results/selection.json` trước khi chạy
test. Hai lỗi test advanced được giữ nguyên, không sửa test hay rerun để tăng điểm.
