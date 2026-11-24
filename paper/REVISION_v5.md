# Manuscript v5 — 17/11/2026

## Phạm vi đã chốt

Dùng nguyên bộ thực nghiệm đã có. Không chạy thêm pretrained inference, training, Kaggle/Colab hoặc GPU. Các thay đổi là sửa phân tích, tổng hợp lại kết quả trên CPU, viết bản thảo mới và xuất PDF.

- Bản đọc: `output/pdf/MEMLEARN_manuscript_v5.pdf`.
- Bản chỉnh sửa: `paper/DRAFT_v5.md`.
- Tựa đề: **Binding Makes Rules Addressable: Key-Mediated Retrieval of In-Context Rules**.
- Bản v4 và release v2.3.0 giữ nguyên để đối chiếu lịch sử.

## Các thay đổi khoa học

1. Sửa pairing: giữ framing/distractor/control cùng các yếu tố cố định, chỉ bỏ yếu tố đang so sánh. Từ chối khóa trùng để tránh ghi đè âm thầm. Ba regression test đã pass.
2. Chốt **13 mô hình**, 7 họ nếu gộp hai thế hệ Qwen. Bảng t20 cho thấy READ > APPLY ở cả 13 point estimate, nhưng CI loại trừ 0 ở **11/13**. Phi và Llama-3.2 không đạt tiêu chí CI này. Bf > rf cho READ có CI loại trừ 0 ở **13/13**.
3. Trọng tâm dương là receiver restoration qua key và reader heads. Giữ riêng kết quả APPLY: P6a chỉ đạt 2/5, nên không mô tả một lời giải đầy đủ cho composition.
4. Kiểm lại điều kiện sender P5a trên số chưa làm tròn: chỉ Qwen3-4B đạt ngưỡng trong bốn model ban đầu. Deficit của Qwen3-8B/Phi/Mistral lần lượt là 0.1250/0.1458/0.1042. Vì vậy không gọi kết luận attention-vs-MLP đa mô hình là confirmatory.
5. Giữ kết quả adapter âm và diễn đạt transfer theo order là không hoàn chỉnh, có khác biệt giữa method/model. Không tuyên bố một kiến trúc mới thắng LoRA.
6. Thu hẹp claim về prose; ghi rõ ngoại lệ, floor effects, và CI chứa 0 không chứng minh tương đương.
7. Bổ sung related work gần cơ chế: binding IDs, mixing mechanisms, retrieval/induction heads, Oh & Demberg, Wang et al., Gao & Gao. Các nhận định về Guo được đối chiếu với toàn văn; phân biệt thứ tự trong thí nghiệm natural-language và symbolic.

## Artifact và kiểm chứng

- `paper/analysis.py` dựng lại **36 bảng**; 33 bảng cũ được cập nhật và thêm t20/t21/t22 từ receipt cũ.
- t15 bổ sung CI cho K-vs-V ở READ, K+V-vs-K và V-vs-blocked. Không coi V-only có effect nhỏ là bằng chứng values không có vai trò.
- Các bảng Markdown được sửa định dạng ký tự pipe và cột head knockout để tránh sai cấu trúc bảng.
- `paper/build_manuscript.py` dựng 2 hình giải thích từ kết quả đã lưu và xuất PDF có 17 mục tài liệu tham khảo.
- `paper/verify_manuscript.py` đối chiếu bốn bảng trong manuscript với CSV phân tích và lưu `paper/manuscript_validation.json`.
- PDF được render bằng Poppler và kiểm tra trực quan toàn bộ trang. Bản hiện tại dùng layout học thuật một cột, chưa áp template của venue.

Môi trường dựng PDF trong phiên này: Python 3.13; phân tích chỉ cần thư viện CPU. Các test mới về paper không cần torch. Bộ test mô hình 128 test có kết quả Windows được ghi trong báo cáo audit trước đó; không tuyên bố vừa chạy lại toàn bộ suite sau revision chỉ chỉnh paper này.

## Trước khi gửi bài

Cần bổ sung tác giả/affiliation và áp template của venue đã chọn. Thông tin này chưa được cung cấp nên PDF không tự điền danh tính. Raw archive vẫn được mô tả là phân phối riêng; không tuyên bố đã có link dữ liệu công khai hoặc pre-registration công khai có timestamp. Các công việc còn lại này không yêu cầu GPU mới.
