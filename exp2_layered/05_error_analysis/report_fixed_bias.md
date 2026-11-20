# Bước 05 — Sai lệch cố định của mạng gắn với tham số da nào?

Chỉ phân tích. Không huấn luyện lại hay sửa mô hình nào, và không đọc locked_test.

- Script: `exp2_layered/05_error_analysis/code/bias_analysis.py`
- Kết quả: `exp2_layered/05_error_analysis/results/fixed_bias/analysis.json` và `bias.npz`

## Cách làm

**Dự đoán**
- Mô hình hetero (trung bình 3 seed) chạy trên tập train (300 da, 1500 mạch) và validation_ext (90 da, 450 mạch).
- Chỉ có nhiễu MC, mỗi mạch dịch ngang 8 lần.
- Sai lệch mỗi mạch là trung bình của sai số qua 8 lần dịch.

**Phân tích**
- So sai số trên train với validation_ext.
- Tương quan của sai lệch với từng tham số (thang log cho μ và melanin; chuẩn hóa theo train).
- Mô hình hóa sai lệch:
  - hồi quy tuyến tính;
  - hồi quy ridge bậc hai (bình phương và tích từng cặp), λ chọn bằng CV 5 lớp theo làn da.
  - Cả hai chỉ fit trên **sai lệch của train**, rồi kiểm trên validation_ext.
- Đo tầm quan trọng của từng tham số bằng cách hoán vị tham số đó trên validation_ext.
- ICC: mức tương quan sai lệch giữa các mạch cùng một làn da.

## Kết quả

| | train | validation_ext |
|---|---|---|
| RMSE (mm) | 0,203 | 0,204 |
| RMS sai lệch cố định | 0,203 | 0,203 |
| Tản theo vị trí dịch | 0,017 | 0,014 |

1. **Không phải học thuộc.** Mạng sai trên chính dữ liệu huấn luyện y như trên dữ liệu mới. Đây là hiện tượng **học chưa tới (underfit)**. Điều này khớp với kết quả trước: thêm dữ liệu không giúp.

2. **Sai lệch là một hàm có hệ thống của tham số da.**
   - Hàm bậc hai fit trên train giải thích **65 %** phương sai sai lệch của validation_ext (CV trên train: 64 %; tuyến tính: 51 %).
   - Bỏ phần dự đoán được thì RMS sai lệch giảm từ 0,203 xuống 0,119 mm. Con số này gần cận bước 05 (giới hạn 14 tham số) (~0,11 mm).

3. **Tham số chính là độ tán xạ của lớp bì (derm_musp).**
   - Tương quan với sai lệch: r = 0,53 trên validation_ext, 0,49 trên train.
   - Độ sâu kèm derm_musp đã giải thích 40 % (riêng độ sâu: 13 %).
   - Hướng sai: bì tán xạ mạnh hơn thì mạng đoán **sâu hơn** thật, khoảng +0,09 mm cho mỗi độ lệch chuẩn. Ý nghĩa vật lý: bì tán xạ mạnh làm bóng mạch nhòe, trông như mạch nằm sâu hơn.
   - Các tham số tiếp theo, xếp theo mức giảm R² khi hoán vị:

     | Tham số | Mức giảm R² | Hướng |
     |---|---|---|
     | derm_musp | 0,50 | |
     | độ sâu | 0,31 | |
     | fat_musp | 0,23 | ngược chiều, r = −0,36 |
     | dermis_mm | 0,16 | |
     | fat_mua | 0,09 | |
     | mua của máu | 0,09 | |

   - Melanin, đường kính và các tham số của lớp cơ gần như không ảnh hưởng.

4. **Sai lệch còn phụ thuộc độ sâu.**

   | Dải độ sâu | Lệch trung bình |
   |---|---|
   | 3,5–4 mm | +0,11 mm |
   | 4,5–5 mm | −0,20 mm (co về biên 5 mm) |

5. **ICC giữa các mạch cùng da = 0,50.** Một nửa sai lệch là chung cho cả làn da, khớp với việc tham số lớp bì quyết định phần lớn.

## Diễn giải

**Kết hợp với bước 05 (giới hạn 14 tham số):**
- Ảnh có đủ thông tin để tách độ sâu khỏi độ tán xạ của bì. Ở bước 05 (giới hạn 14 tham số), tương quan hậu nghiệm giữa độ sâu và derm_musp chỉ 0,19.
- Nhưng mạng chưa học được cách bù trừ đó, ngay cả trên dữ liệu huấn luyện.
- Vấn đề nằm ở **quá trình fit** (mục tiêu huấn luyện, dải nhiễu tăng cường, dung lượng hoặc tối ưu hóa), không nằm ở lượng dữ liệu hay thông tin trong ảnh.

**Giả thuyết có thể kiểm chứng.** Chỉ cần huấn luyện trên train và chọn trên validation_ext; không cần mô phỏng MC.
- **(a) Nhiễu tăng cường khi huấn luyện** (σ log-đều 0,001–0,01). Ở σ cao không còn thông tin, nên mạng học một ánh xạ "thỏa hiệp" cho mọi mức nhiễu. Thử: huấn luyện với σ thấp và hẹp, rồi xem sai số trên train ở σ = 0 có giảm không.
- **(b) Dung lượng hoặc thời gian huấn luyện.** Thử: mạng rộng hơn, huấn luyện lâu hơn; xem sai số trên train có giảm không.
- **(c) Cho mạng biết mức nhiễu** (thêm σ làm đầu vào).

**Lưu ý**
- Hồi quy bậc hai chỉ xác định phần sai lệch giải thích được. Đó chưa phải nguyên nhân.
- Kết quả được tính ở σ = 0, tức chỉ có nhiễu MC.
