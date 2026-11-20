# Bước 02 — Giới hạn thông tin và tham số gây nhiễu (850 nm, mô hình đồng nhất + kênh máu)

Kết quả nằm trong `results/` của thư mục này. Không có tệp nào của gói luận văn bị sửa. V1 là bộ đã mở, nên mọi con số tính trên V1 chỉ mang tính khám phá.

## bước 01 — Kiểm mô hình thuận với Farrell 1992

Môi trường đồng nhất μa 0,01 mm⁻¹, μs′ 1,0 mm⁻¹, n 1,35; chùm bút chì; 60 triệu photon. Tệp: `exp2_layered/02_forward_model/results/nuisance/analysis.json`.

- Chuẩn hóa và đơn vị đúng: hệ số tỉ lệ tốt nhất MC/Farrell là 1,065.
- Ở xa (ρ 8–10 mm, tức 8–10 khoảng tự do vận chuyển), MC và Farrell lệch nhau 3,5–6 %.
- Ở gần nguồn (ρ 1–1,4 mm) MC thấp hơn Farrell 16–27 %. Lời giải khuếch tán không còn đúng trong vùng dưới khoảng 2 khoảng tự do, nên mức lệch này là dự kiến.
- **Chưa đạt trọn:** ở ρ 2,6–5 mm MC cao hơn Farrell 13–18 %. Mức này lớn hơn mức thường chấp nhận. Có thể do giới hạn của mô hình lưỡng cực khi chiết suất không khớp, cũng có thể do cách MCX tính phản xạ trên voxel bề mặt. Cần một phép so độc lập: so với bảng chuẩn MCML (Wang & Jacques), hoặc chạy lại với n = 1.

### bước 01 — Kiểm độc lập (đã giải quyết)

- **Mã đối chiếu:** Monte Carlo trên CPU kiểu MCML, viết bằng numpy (`exp2_layered/01_simulator_checks/code/cpu_mc_reference.py`). Mã này khác MCX về cách tính hấp thụ, cách xử lý Fresnel và cách tính điểm.
- **Kiểm mã với lời giải chính xác:** so với hàm H của Chandrasekhar cho nửa không gian tán xạ đẳng hướng, n = 1.
  - Albedo 0,9: R = 0,4151 (CPU MC) so với 0,4150 (lý thuyết).
  - Albedo 0,99: R = 0,7524 so với 0,7527.
- **Kết quả với môi trường bước 01** (`exp2_layered/01_simulator_checks/results/cpu_vs_gpu/t1b_summary.json`):
  - MCX/CPU theo bán kính 1–10 mm: 0,983–1,007 khi n = 1,35, và cũng 0,983–1,007 khi n ≈ 1.
  - Phản xạ toàn phần: 0,7484 (MCX) so với 0,7490 (CPU), n ≈ 1.
  - Farrell so với CPU: lệch tới −16 % ở 3–4 mm khi n = 1,35, và +10–18 % ở 1–3 mm khi n = 1.
- **Kết luận:** MCX đúng trong khoảng ±1 %. Phần lệch 13–18 % ở bước 01 là sai số của xấp xỉ khuếch tán lưỡng cực, không phải lỗi mô hình thuận. bước 01 đạt.

## bước 02 sơ bộ — Lượng thông tin khi chỉ μa chưa biết (dữ liệu V1)

Tệp: `exp2_layered/02_forward_model/results/crlb_preliminary/`.

- Hệ số phạt của μa chưa biết lên CRLB độ sâu là 1,0–3,2.
- CRLB thấp hơn sai số của F rất nhiều.
- Spearman giữa CRLB và sai số của F chỉ 0,39. Vậy sai số của F chủ yếu không đến từ giới hạn thông tin.

## B3 — Khớp mô hình vật lý (bảng tra giai đoạn 4, cùng ước lượng μa)

Tệp: `exp2_layered/04_training/results/b3_physics_fit/`. MAE (mm) trên V1:

| Nhóm | B3, ước lượng μa | F đã khóa |
| --- | --- | --- |
| μa chưa học | 0,072 | 0,159 |
| Toàn bộ | 0,057 | 0,166 |

## bước 04 thử — F huấn luyện đủ (tối đa 400 epoch, chọn theo validation)

Tệp: `exp2_layered/04_training/results/longer_training/`. MAE (mm) trên V1:

| Nhóm | Seed 42 / 43 / 44 | F đã khóa |
| --- | --- | --- |
| Toàn bộ | 0,050 / 0,048 / 0,057 | 0,166 |
| μa chưa học | 0,055 / 0,057 / 0,070 | 0,159 |

Kết luận: nút thắt chính của F là huấn luyện chưa đủ.

## bước 02 — CRLB độ sâu khi thêm tham số nhiễu (Jacobian MC)

Tệp: `exp2_layered/02_forward_model/results/nuisance/analysis.json`.

Các bộ tham số lồng nhau:

- S1: chỉ độ sâu.
- S2: thêm μa kênh.
- S3: thêm dịch ngang.
- S4: thêm μs′ nền.
- S5: thêm đường kính kênh.
- S6: thêm μa nền.

| z (mm) | μa kênh | Hệ số phạt S2 → S6 | √CRLB S6 ở σ = 0,003 | √CRLB S6 ở σ = 0,01 |
| --- | --- | --- | --- | --- |
| 1,4 | 0,15 | 2,3 → 5,6 | 0,029 | 0,10 |
| 2,8 | 0,15 | 2,9 → 4,1 | 0,046 | 0,15 |
| 3,8 | 0,15 | 2,5 → 2,9 | 0,050 | 0,17 |
| 1,4 | 0,508 | 2,4 → 5,7 | 0,021 | 0,07 |
| 2,8 | 0,508 | 3,1 → 4,2 | 0,033 | 0,11 |
| 3,8 | 0,508 | 3,0 → 3,4 | 0,041 | 0,14 |

Hệ số phạt khi thêm từng tham số riêng lẻ:

- μa kênh: 2,3–3,1, lớn nhất.
- Đường kính: 2,0–2,5.
- μa nền: 1,5–2,5.
- μs′ nền: 1,6–2,3.
- Dịch ngang: 1,0, tức không gây hại.

**Quyết định ở cổng:** độ sâu vẫn xác định được khi cả năm tham số nhiễu đều chưa biết, với điều kiện nhiễu mỗi bin σ ≲ 0,003. Đến σ ≈ 0,01 thì sai số ở vùng sâu chạm mức 0,1–0,17 mm.

Vì vậy đại lượng quyết định là **σ đo được trên phantom/camera**. Đây là phép đo đầu tiên cần làm khi có phantom.

## Giới hạn

1. CRLB ở đây là cận cục bộ, tuyến tính, với nhiễu trắng; nó không tính sai lệch mô hình.
2. Nhiễu MC trong sai phân hữu hạn làm CRLB hơi lạc quan.
3. Mô hình vẫn là một lớp đồng nhất. Rủi ro lớn nhất cho bài báo là da nhiều lớp (melanin, mỡ) ở bước 02, chưa được kiểm.

## Bước tiếp theo đề xuất

- **bước 01:** so độc lập với MCML hoặc chạy n = 1 để giải thích phần lệch 13–18 %.
- **bước 02:** mô hình da nhiều lớp.
  - Trước hết bổ sung bảng tham số: tính chất quang của mỡ, độ dày bì và mỡ.
  - Lập Jacobian cho độ dày biểu bì, melanin và độ dày mỡ.
- **bước 03–bước 04:** dữ liệu huấn luyện ngẫu nhiên hóa đủ năm tham số nhiễu; mô hình ước lượng đồng thời. So F (huấn luyện đủ) với B3 trên **một bộ khóa mới**.
- **Luận văn:** đưa B3 vào nhóm đối chứng (S1). Phiên bản F huấn luyện đủ chỉ được chấm trên bộ kiểm chứng mới.
