# Bước 04 — F_layered trên tập validation

`locked_test` **chưa mở**. Mọi số dưới đây tính trên tập validation của bước 03: 30 cấu hình da chưa từng dùng để huấn luyện, 150 tĩnh mạch.

Mã: `exp2_layered/04_training/code/train_layered_cnn.py` (`build`, `train <seed>`, `eval`). Kết quả: `exp2_layered/04_training/results/layered_cnn/` (`validation_eval.json`, `validation_predictions.npz`, `models/`).

## Cách làm

**Dữ liệu huấn luyện**
- 1500 tĩnh mạch của tập train, mỗi mạch tạo 24 bản → 36 000 mẫu.
- Mỗi bản: dịch ngang U(−2, 2) mm, gộp 128 bin trên [−16, 16] mm, thêm nhiễu camera σ_add log-đều trong [0,001; 0,01].
- Cách đọc giống giai đoạn 4: profile = trung bình theo y / N0 vô hướng.

**Huấn luyện**
- Mô hình DepthCNN của luận văn, giữ nguyên.
- Adam 1e-3, batch 64, hàm mất mát L1, tối đa 400 epoch, dừng sớm sau 40 epoch.
- Chọn checkpoint theo validation: 150 mạch × 8 bản, cùng kiểu tăng cường với train.
- Chạy CPU, chế độ tất định, ba seed 42 / 43 / 44.

| Seed | Epoch tốt nhất | Epoch dừng | MAE validation lúc chọn | Thời gian |
| --- | --- | --- | --- | --- |
| 42 | 52 | 92 | 0,190 mm | 453 s |
| 43 | 66 | 106 | 0,199 mm | 502 s |
| 44 | 95 | 135 | 0,187 mm | 573 s |

**Nhiễu dùng khi chấm**
- Bản thân profile bước 03 (12 triệu photon) đã có nhiễu MC khoảng **0,0049 mỗi bin**, cao hơn mức σ ≈ 0,003 mà bước 02 cho là cần.
- Vì vậy chấm theo **nhiễu tổng mỗi bin**, ba mức:
  - chỉ nhiễu MC (≈ 0,005);
  - 0,007;
  - 0,010.
- Mỗi mạch có 4 bản dịch ngang khác nhau. B3 tốn thời gian hơn nên chỉ chấm 1 bản mỗi mạch.

## Kết quả

MAE (mm) trên 150 tĩnh mạch validation. Với các mô hình CNN, số ghi là trung bình của 3 seed.

| Phương pháp | Nhiễu MC (~0,005) | Tổng 0,007 | Tổng 0,010 | Bias ở mức MC |
| --- | --- | --- | --- | --- |
| **F_layered** (trung bình 3 seed) | **0,178** | 0,191 | 0,219 | ≈ 0 (−0,02 … +0,04) |
| F_layered, gộp 3 seed (ensemble) | 0,171 | 0,183 | 0,209 | +0,01 |
| F huấn luyện lâu, mô hình đồng nhất | 0,656 | 0,666 | 0,688 | +0,46 |
| F đã khóa (luận văn) | 1,187 | 1,189 | 1,196 | +0,91 (seed 42) |
| B3 đồng nhất (khớp vật lý) | 1,293* | 1,299* | 1,282* | +1,29* |
| Hằng số = trung bình độ sâu tập train | 0,883 | 0,883 | 0,883 | −0,09 |

\* B3 chạm cận trên 5 mm ở 36–38 % số mạch, nên MAE thật của B3 còn lớn hơn số ghi.

### Theo độ sâu và màu da (nhiễu MC; F_layered tính theo seed 42 / 43 / 44)

| Nhóm | F_layered | F đã khóa (seed 42) | B3 đồng nhất |
| --- | --- | --- | --- |
| Độ sâu < 2 mm | 0,12 / 0,13 / 0,12 | 0,93 | 1,26 |
| 2–3 mm | 0,14 / 0,16 / 0,14 | 1,58 | 1,65 |
| 3–4 mm | 0,25 / 0,24 / 0,21 | 0,87 | 1,40 |
| ≥ 4 mm | 0,24 / 0,25 / 0,25 | 0,31† | 0,57† |
| Melanin < 5 % | 0,19 / 0,19 / 0,18 | 1,14 | 1,36 |
| Melanin ≥ 15 % | 0,19 / 0,20 / 0,17 | 0,92 | 1,25 |

† Ở nhóm ≥ 4 mm, sai số của mô hình đồng nhất trông nhỏ vì dự đoán bị chặn ở cận 5 mm. Thực chất đây vẫn là lỗi đoán quá sâu.

## Nhận xét

1. **RQ3 được ủng hộ trên validation.** Huấn luyện trên da nhiều lớp đưa MAE từ 1,0–1,3 mm (mô hình đồng nhất) xuống **0,17–0,19 mm**, và bias gần 0. So với dự đoán hằng số, sai số giảm khoảng 5 lần.
2. **Mô hình bền với màu da.** Da sáng và da sẫm có MAE như nhau (0,18–0,20 mm). Điều này khớp với bước 02: chuẩn hóa theo N0 khử được phần lớn ảnh hưởng của melanin.
3. **Vùng khó là độ sâu ≥ 3 mm.** MAE ở đây khoảng 0,21–0,29 mm, gấp đôi vùng nông. Đây là nơi độ tương phản chỉ còn 1–5 %.
4. **Huấn luyện lâu trên mô hình đồng nhất không cứu được.** F_long chỉ giảm sai số xuống 0,66 mm, bias vẫn +0,46 mm. Nút thắt ở đây là **dữ liệu sai mô hình**, không phải huấn luyện chưa đủ. Tình huống này khác với kết quả trên V1, nơi môi trường thật sự đồng nhất.
5. **Còn cách giới hạn lý thuyết.** CRLB cục bộ ở bước 02 (một điểm, D = 3 mm) là 0,045 mm ở σ = 0,003. Đổi tỉ lệ sang σ ≈ 0,005 được khoảng 0,075 mm. F_layered đang khoảng 0,17 mm, tức gấp chừng 2 lần.
   - Phép so này chỉ là định hướng. Tập validation trải rộng hơn bước 02 về đường kính, μa và độ sâu tới 5 mm, nên không so trực tiếp được.
   - Nguyên nhân có thể:
     - chỉ có 300 cấu hình da độc lập;
     - mô hình nhỏ;
     - learning rate cố định, dừng sớm ở epoch 52–95.

## Giới hạn

- Mọi kết quả là mô phỏng. Chưa có phantom hay dữ liệu người.
- Nhiễu MC của bước 03 (≈ 0,005) cao hơn mức camera tốt. Locked_test (60 triệu photon) ít nhiễu hơn và sẽ được chấm ở cùng các mức nhiễu tổng.
- B3 chỉ dùng bảng tra đồng nhất có sẵn. Chưa có phép khớp vật lý dùng mô hình da nhiều lớp.

## Quyết định cần đưa ra trước khi mở locked_test

- **Phương án A:** khóa F_layered hiện tại (công thức đã đăng ký trong T4_thiet_ke.md) và chấm locked_test một lần. Đây là kết quả chính.
- **Phương án B:** cải thiện trước trên validation (thêm dữ liệu, lịch learning rate, mô hình lớn hơn), rồi mới khóa.
- Dù chọn phương án nào, locked_test cũng chỉ chấm được **một** mô hình cuối. Cải tiến làm sau khi đã mở sẽ cần một bộ test khóa mới (khoảng 2,5 giờ GPU).
