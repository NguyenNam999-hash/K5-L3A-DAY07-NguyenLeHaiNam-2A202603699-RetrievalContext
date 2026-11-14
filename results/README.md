# Results

| Path | Nội dung |
|---|---|
| `retrieval/summary_dev.json`, `retrieval/summary_test.json` | So sánh 7 cấu hình retrieval |
| `retrieval/<config>/<split>/traces.json` | Top-k, score/rank, gold hit, context, latency, cost |
| `agent/<config>/<split>/traces.json` | Tool call, query, kết quả, context, answer, citation, usage, lỗi |
| `agent/<config>/<split>/reviews.json` | Nhãn nháp grounded/citation/abstention/context |
| `agent/<config>/<split>/metrics.json` | Tổng hợp nhãn |
| `selection.json` | Cấu hình và hash test chốt trước khi chạy test |
| `test_execution.json` | Snapshot kết quả test sau khi chốt |
| `analysis.md` | Bảng kết quả, lỗi dev/test, đánh đổi, giới hạn |

Nhãn trong `reviews.json` do AI soạn nháp từ trace; sinh viên cần đọc lại trước khi coi là
nhãn người. Các file `.env`, vector snapshot, PDF advanced và index tái tạo được không
commit.
