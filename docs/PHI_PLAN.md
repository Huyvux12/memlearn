# Kiểm giả thuyết gốc trên Phi-3.5 và cấu hình 1 layer trên Qwen — tiền đăng ký (22/10/2026)

Viết trước khi chạy. Bối cảnh:
- `runs/e1_phi35_mini_colab/ANALYSIS.md` cho thấy Phi-3.5 đóng băng áp quy tắc ngoài phạm vi ở book_first. Đây là phát hiện sau dữ liệu, nên cần xác nhận trên dữ liệu mới.
- `runs/e3_colab/ANALYSIS.md` cho thấy xếp hạng rò rỉ do adaptation trên Qwen3-4B (4 layer) là VECTOR_PAIR < SEPARABLE < LORA_QKVO.

## P — Phi-3.5-mini-instruct

**Thiết lập:**
- **Model:** `microsoft/Phi-3.5-mini-instruct`, đóng băng, fp16, 32 layer.
- **Adapter:** ở layer {5, 11, 16, 21}, cùng độ sâu tương đối với {6, 12, 18, 24} trên Qwen 36 layer.
- **Dữ liệu:** `data_seed = 23`, **dữ liệu mới**, không trùng với lượt E1 có phát hiện ban đầu.
- **Train:** 256 bước, lr 5e-4, số tham số được khớp với VECTOR_PAIR. Item train lấy từ book_first, gồm mọi mode, framing và distractor.
- **Seed:** 17, 29, 43.
- **Method:** FULL, QUERY, SEPARABLE, VECTOR_PAIR, LORA_QKVO (LoRA trên `qkv_proj` gộp và `o_proj`).
- **Đánh giá:** neutral framing × {none, q} × {book_first, rule_first}; 12 + 12 bundle; mode READ_DEC, READ_RULE, APPLY_P, NONAPPLY; probe log-prob 4 ứng viên.

**P0, xác nhận lỗi mục tiêu.** FULL có rò rỉ NONAPPLY (chọn P(base)) ≥ 10 % ở book_first (n = 48: 24 bundle × 2 distractor). Nếu không đạt, phần P chỉ còn là một lượt so sánh adapter thông thường, và giả thuyết gốc không được kiểm ở đây.

**Kết quả chính.** Tỷ lệ rò rỉ ở book_first sau adaptation, gộp 3 seed (n = 144 mỗi method).

**Dự đoán của giả thuyết gốc**, theo đó conditioning theo cặp trước aggregation giúp dùng thông tin có chọn lọc:
1. Rò rỉ của VECTOR_PAIR < LORA_QKVO và < QUERY. Contrast ghép cặp theo item, bootstrap theo cụm (seed × bundle); CI95 phải loại 0.
2. Đồng thời VECTOR_PAIR giữ được APPLY_P ở book_first và READ_DEC/READ_RULE ≥ FULL − 0,02.

**Quyết định:**

| Kết quả | Kết luận |
|:---|:---|
| Cả hai dự đoán đúng | **Ủng hộ giả thuyết gốc** trên model có lỗi mục tiêu. Đây là kết quả trung tâm của paper |
| Mọi adapter loại được rò rỉ như nhau | Giả thuyết không cần thiết: adaptation bất kỳ đều sửa được lỗi phạm vi |
| VECTOR_PAIR rò rỉ ít nhưng mất APPLY | Trade-off, không ủng hộ trọn vẹn |

Thứ cấp: APPLY_P ở rule_first, JSS-v2 ở book_first, và median margin NONAPPLY.

## Q — Qwen3-4B, cấu hình 1 layer (nhóm A, 3 seed)

**Thiết lập:** layer giữa (18), lr 2e-3, 256 bước, `data_seed = 17`. Method: QUERY, SEPARABLE, VECTOR_PAIR, LORA_V. Seed 17, 29, 43.

- Seed 17 lặp lại đúng cấu hình nhóm A, nên đồng thời kiểm **khả năng tái lập trên GPU khác** (T4 so với G4).

**Câu hỏi:** ở 1 layer với lr cao, xếp hạng rò rỉ có khác với 4 layer không? Nhóm A, 1 seed: VECTOR_PAIR 15/96, LORA_V 15/96, QUERY 7/96, SEPARABLE 0/96.

**Kết quả:** rò rỉ gộp 3 seed (n = 288 NONAPPLY mỗi method). Báo cáo xếp hạng và CI.

**Diễn giải đã chốt trước:**
- Nếu VECTOR_PAIR rò rỉ nhiều nhất ở 1 layer nhưng ít nhất ở 4 layer, claim C5 phải viết là *phụ thuộc cấu hình*.
- Không được chọn cấu hình theo kết quả.

## Tối ưu tài nguyên (không đổi thuật toán)

- **Chạy song song:** các stage chạy song song trên một GPU, mỗi stage một process. Model được tải trước một lần vào cache dùng chung, và test chỉ chạy một lần.
- **Chunk:** `chunk = 64` cho gate theo cặp. Kết quả không phụ thuộc chunk; tính chất này đã có test (`test_query_and_separable_exact_algebra_and_chunk_invariance`).
- Batch train vẫn là 1, giống E3, nên mọi kết quả so sánh được với nhau.
