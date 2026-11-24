# Hướng A — Kế hoạch paper phân tích và cơ chế (chốt 22/10/2026)

## Câu chuyện

> Một LLM đóng băng **tách phạm vi quy tắc rất chắc**, nhưng **không ghép được hai nguồn trong ngữ cảnh khi phải áp dụng quy tắc**. Khả năng ghép phụ thuộc **thứ tự nguồn**, qua một cơ chế mà ở đó *source đứng sau đọc source đứng trước*. Adaptation (LoRA và các adapter attention) sửa được lỗi trong phân phối nhưng **không tổng quát hóa qua thứ tự nguồn**. Khi train mạnh, adaptation còn **gây rò rỉ phạm vi**.

Venue mục tiêu: BlackboxNLP (EMNLP workshop) hoặc Findings. Có thể nhắm cao hơn nếu kết quả lặp lại trên nhiều model và phần cơ chế đủ sắc.

## Claim và bằng chứng cần có

| # | Claim | Đã có (1 lượt) | Còn thiếu |
|:---|:---|:---|:---|
| C1 | Model nền không áp quy tắc ngoài phạm vi, kể cả dưới áp lực | Qwen3-4B: 0/384 | ≥ 2 model khác |
| C2 | Lỗi ghép hai nguồn khi áp dụng, có phân tách read/apply và bất đối xứng thứ tự | Qwen3-4B: APPLY 0,50; READ_RULE 0,84; bf 0,60 so với rf 0,27 | ≥ 2 model khác |
| C3 | Source đứng sau phải đọc source đứng trước (knockout nhân quả) | Có, nhưng block phá cả READ_RULE | Knockout **hẹp hơn** (theo dải layer, chỉ token key hoặc chỉ token value) để tách phép ghép khỏi việc đọc rule |
| C4 | Adaptation không tổng quát hóa qua thứ tự | 2 lượt sàng lọc, 1 seed | **3 seed**; thêm control "train cả hai thứ tự" làm cận trên |
| C5 | Rò rỉ phạm vi phụ thuộc cường độ adaptation | lr 2e-3 có rò rỉ, lr 5e-4 không | Quét lr {5e-4, 1e-3, 2e-3}, 3 seed |

Kết quả âm của VECTOR_PAIR (giả thuyết ban đầu) được báo cáo trung thực, như một lớp toán tử nằm trong C4/C5.

## Thí nghiệm

### E2 — Knockout hẹp (Qwen3-4B, khoảng 6 phút T4)

Dùng thứ tự book_first, không có Q. Query luôn là token của rule. Các variant:

| Variant | Key bị chặn | Layer bị chặn |
|:---|:---|:---|
| (tham chiếu) | không chặn | — |
| (đối chứng) | sham | mọi layer |
| (đối chứng) | toàn bộ codebook | mọi layer |
| dải layer | toàn bộ codebook | chỉ đầu (0–11), chỉ giữa (12–23), hoặc chỉ cuối (24–35) |
| chỉ key | chỉ token key `x..` của codebook | mọi layer |
| chỉ value | chỉ token value `z..` của codebook | mọi layer |

**Dự đoán chốt trước:**
- Có ít nhất một variant hẹp làm APPLY_P giảm ≥ 0,20 so với tham chiếu trong khi READ_RULE giảm ≤ 0,10. Nếu đúng, ta có bằng chứng về một bước *ghép* tách khỏi bước *đọc*.
- Nếu không variant nào đạt, claim C3 giữ ở mức "đọc rule cần tích hợp cross-source", không tách được phép ghép.
- Dự đoán hướng: chặn *chỉ key* phá phép ghép nhiều hơn phá việc đọc rule. Lý do: rule cần biết `x` nào dẫn tới `z` này, còn việc đọc rule chỉ cần khớp `z`.

### E1 — Lặp lại trên model khác (probe log-prob và knockout cơ bản)

- Model không cần token HF: **Qwen3-1.7B** (cùng họ, khác cỡ), **Qwen2.5-3B-Instruct** (thế hệ khác), **Phi-3.5-mini-instruct** (kiến trúc khác).
- Code probe và knockout được tổng quát hóa để chạy với bất kỳ kiến trúc HF nào có `eager_attention_forward`.
- Mỗi model khoảng 15 phút T4.

### E3 — Adaptation và thứ tự nguồn (Qwen3-4B, 3 seed)

- Methods: FULL, LORA_QKVO, SEPARABLE, VECTOR_PAIR. Cấu hình như lượt revisit: 4 layer, lr 5e-4.
- Ba chế độ train:
  1. chỉ book_first, test trên rule_first (kiểm C4);
  2. **cả hai thứ tự** (cận trên; cho biết lỗi là do capacity hay do không tổng quát hóa);
  3. quét lr {5e-4, 2e-3} cho C5.
- Mỗi seed của mỗi chế độ khoảng 20 phút T4. Đây là phần nặng nhất, nên chạy trên Colab Pro.

## Thứ tự làm

1. **E2.** Rẻ và trả lời câu hỏi cơ chế còn mở.
2. **Tổng quát hóa code cho nhiều kiến trúc**, sau đó chạy E1.
3. **E3** trên Colab Pro.
4. **Viết bản thảo:** related work dựa trên báo cáo novelty (Guo 2025, Premise Order, Lost-in-order, Hopping Too Late, Two-Hop Curse).
