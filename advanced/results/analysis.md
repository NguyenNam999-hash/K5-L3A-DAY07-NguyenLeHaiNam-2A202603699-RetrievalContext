# Advanced — phân tích kết quả

## 1. Nguồn và index

- **Tài liệu:** World Bank, *Global Economic Prospects, June 2024*, 220 trang, CC BY 3.0 IGO.
- **Manifest/hash:** `advanced/source/manifest.json`; PDF không commit, tải bằng
  `advanced/scripts/fetch_document.py`.
- **Extraction:** 220/220 trang; 194 trang có text; 26 trang bìa/mục lục/trống.
  31 trang có bảng, 72 trang có figure; 38 bảng, 72 figure (65 chart + 7 image).
- **Chunk:** 359 (249 text, 38 table, 65 chart, 7 image); mỗi chunk giữ page, content_type,
  caption/bbox và metadata.
- **Index:** `BAAI/bge-base-en-v1.5` 768 chiều, pgvector bảng `advanced_chunks`, cosine
  exact scan. Embedding 359 chunk mất 153,94 giây.
- **Vision:** 66 call, cache tại `advanced/cache/`; chi phí list khoảng $0,054 cho lần đầu.
  Sáu map chapter opener được mô tả thủ công vì model gọi map nhỏ là decorative.

## 2. Xử lý bảng, chart, ảnh

- **Bảng:** dùng bbox từng cell + `page.get_text("text", clip=cell)`, không dùng
  `table.extract()` vì cách đó làm mất dấu thập phân (`6.3` thành `63 .`). 38 bảng được
  lưu thành Markdown + cell rows; 6 bảng test được đối chiếu thủ công với trang render.
- **Chart:** render trang, gửi vision model mô tả trục/đơn vị/series/giá trị; description
  được index như chunk `chart`. 5 câu chart test được đọc lại trên ảnh gốc.
- **Map/ảnh:** 6 locator map chapter opener + barcode back cover. Map descriptions do
  người viết từ ảnh render, không dựa vào suy đoán của model.
- **Text:** chunk theo paragraph, giữ heading/page; không đưa gold vào chunk.

## 3. Benchmark

Chia trước khi chạy; test hash ghi tại `advanced/results/selection.json`:

| Split | Có đáp án | Thiếu evidence | Tổng |
|---|---:|---:|---:|
| dev | 15 | 3 | 18 |
| test | 24 | 6 | 30 |

Test groups: 6 table, 5 image/map, 5 chart, 4 multi-hop, 4 text, 6 unanswerable.
Mỗi câu answerable có gold `{page, content_type, evidence}`; evidence là substring nguyên
văn của chunk đúng page/type. Câu thiếu evidence có `absence_checks` regex không khớp corpus.
`advanced/scripts/verify_benchmark.py` kiểm tra toàn bộ trước khi chạy.

## 4. Kết quả dev

| Pipeline | Answerable answered | Unanswerable abstained | Errors | Token | Cost |
|---|---:|---:|---:|---:|---:|
| Dense baseline | 12/15 | 3/3 | 0 | 235.434 | $0,0260 |
| **Selected (hybrid + rerank)** | **15/15** | **3/3** | 0 | 1.179.891 | $0,1394 |

Chọn selected vì bao phủ đủ 15/15 answerable và 3/3 abstention; baseline bỏ 3 câu
answerable (ad-t03, ad-c02, ad-x02).

## 5. Kết quả test

| Nhóm | Đạt / tổng | Tỷ lệ | Ghi chú |
|---|---:|---:|---|
| Bảng | 6/6 | 100% | Đúng hàng/cột/năm, giữ đơn vị % |
| Ảnh, map | 5/5 | 100% | Nhận đúng region từ locator map; ISBN từ barcode |
| Biểu đồ | 5/5 | 100% | Đọc đúng 38%, 2024, 14%, xu hướng 2026, 5–6% |
| Multi-hop | 3/4 | 75% | at-m04 bị từ chối do citation không hợp lệ |
| Text | 3/4 | 75% | at-x01 bị từ chối do citation không hợp lệ |
| Thiếu evidence | 6/6 | 100% | Từ chối đúng, không thêm khẳng định |
| **Toàn bộ** | **28/30** | **93,3%** | **Đạt ngưỡng 90% (27/30)** |

- Test latency trung bình 63,3 giây/câu; 2.265.158 token; list cost $0,2534.
- Trace: `advanced/results/agent/selected/test/traces.json`.
- Nhãn pass/fail: `advanced/results/agent/selected/test/reviews.json`.
- Metrics nhóm: `advanced/results/agent/selected/test/metrics.json`.

## 6. Hai lỗi test

### at-x01 — global growth 2025-26

- Câu hỏi: global growth projected for 2025-26?
- Evidence đúng nằm ở `adv-p025-text-000` (“edging up to 2.7 percent in 2025-26”).
- Agent search hai lượt và `get_page`, nhưng final answer không có citation hợp lệ; validator
  từ chối và agent abstain. Đây là lỗi citation/format, không phải thiếu evidence.
- Đã giữ nguyên là fail; không sửa test hay rerun để tăng điểm.

### at-m04 — LIC vs world growth 2024

- Câu hỏi: LIC growth nhanh hơn world bao nhiêu trong 2024?
- Evidence đúng: `adv-p050-t01` (LIC 5.0%) và `adv-p026-t01` (World 2.6%); cả hai đã có
  trong context.
- Agent final answer không có citation hợp lệ; validator từ chối; abstain.
- Đây cũng là lỗi citation, không phải retrieval. Giữ nguyên fail.

**Bài học:** agent cần một bước repair citation mạnh hơn hoặc cho phép citations dạng
nhóm rõ ràng; hiện tại một final answer thiếu citation làm mất một câu đúng.

## 7. Ví dụ đạt

- **at-t03 (table):** Table B1.2.1, Rwanda 2024 = 7.6%; agent cite `adv-p050-t01`; đúng
  hàng/cột/năm, giữ đơn vị %.
- **at-c01 (chart):** Figure 1.1 panel D, EMDEs vulnerable to debt stress ≈ 38%; agent
  cite `adv-p027-f01`; description khớp ảnh.
- **at-i01 (image/map):** map p91 highlight LAC; agent cite `adv-p091-f01`; đúng region.
- **at-m02 (multi-hop):** text p107 (6.6% → 6.2%) + Table 2.5.1 (6.2%); agent cite cả hai;
  câu trả lời kết hợp đúng.

## 8. Giới hạn

- 26 trang không có text; ảnh hưởng chủ yếu bìa/mục lục, không ảnh hưởng benchmark.
- Bảng merged cell/multi-line header vẫn có thể sai; 6 bảng test đã kiểm tra thủ công,
  nhưng 32 bảng còn lại chưa audit từng ô.
- Vision description có thể bỏ sót nhãn nhỏ; 5 chart test đã đối chiếu ảnh, nhưng 60 chart
  còn lại chưa kiểm tra từng giá trị.
- Map descriptions do người viết, không phải model; cần đọc lại khi dùng cho câu khác.
- Test 30 câu, 2 lỗi citation; 93,3% chỉ đúng cho bộ test này, không suy rộng.
- Rerank + agent dùng cùng model, chi phí/latency cao (63 giây/câu test).
- Nhãn pass/fail do AI soạn nháp từ trace; sinh viên cần duyệt lại.
