# Bước 05 — Sai lệch cố định có giảm được bằng cách huấn luyện khác không?

Kế hoạch ghi **trước khi chạy**. Không mô phỏng thêm, chỉ huấn luyện trên CPU.
- Dữ liệu huấn luyện: bước 03 train (300 da).
- Chọn checkpoint: không dùng (lấy trọng số cuối).
- So sánh: validation_ext (90 da).
- locked_test không bị đọc.

## Bối cảnh

- bước 05 (giới hạn 14 tham số): ảnh đủ thông tin cho RMSE khoảng 0,11 mm ở nhiễu thấp, nhưng mạng đạt 0,19–0,20 mm.
- bước 05 (sai lệch cố định): mạng sai bằng nhau trên train và validation_ext (0,203 so với 0,204 mm), tức là học chưa tới. 65 % sai lệch là một hàm của tham số da, chủ yếu là derm_musp.
- Quan sát mới từ lịch sử huấn luyện `hetero_seed42`:
  - MAE trên tập chọn checkpoint dao động mạnh giữa các epoch (0,18–0,33 mm).
  - Checkpoint tốt nhất ở epoch 23–34; huấn luyện dừng sớm ở epoch 83–94, trong khi loss huấn luyện vẫn đang giảm (−1,14 → −1,28).
  - Như vậy mạng có thể chưa hội tụ.

## Biến thể

Tất cả dùng WideCNN với đầu ra Gaussian NLL (giống hetero), 3 seed (42/43/44), và cùng cách huấn luyện:
- 400 epoch cố định;
- LR theo cosine từ 1e-3 xuống 1e-5;
- lấy **trọng số cuối**, không dừng sớm, không chọn checkpoint.

| Mã | Thay đổi so với hetero | Giả thuyết kiểm |
|---|---|---|
| L | Chỉ đổi cách huấn luyện như trên | Học chưa tới do huấn luyện ngắn hoặc chọn checkpoint nhiễu |
| LN | L + nhiễu tăng cường hẹp, log-đều 0,001–0,003 (thay vì 0,001–0,01) | (a) Thỏa hiệp giữa các mức nhiễu |
| LB | L + mạng rộng hơn: 96 kênh thay 48, đầu 128 thay 64 | (b) Thiếu dung lượng |
| LS | L + log σ của nhiễu tăng cường làm đầu vào thêm cho đầu ra; khi đánh giá truyền σ đã biết, σ = 0 dùng 0,001 | (c) Mạng không biết mức nhiễu |

## Đánh giá

Script `retrain_eval.py`, dùng ensemble 3 seed, mỗi mạch 8 vị trí dịch.
- Đo trên train và validation_ext ở σ = 0 (chỉ nhiễu MC), và trên validation_ext ở σ = 0,005 / 0,007 / 0,010.
- Các chỉ số: RMSE, MAE, độ phủ ±1σ/±1,96σ, tương quan giữa sai lệch và derm_musp.

## Tiêu chí (cố định)

**Chỉ số chính:** RMSE trên validation_ext ở σ = 0. Mốc so sánh là hetero = 0,204 mm, tính cùng cách.

1. Một biến thể **sửa được** sai lệch khi RMSE chính giảm ≥ 0,03 mm và RMSE trên train cũng giảm ≥ 0,03 mm.
2. Quy nguyên nhân theo thứ tự:
   - nếu L đạt tiêu chí 1 → nguyên nhân là huấn luyện ngắn hoặc chọn checkpoint;
   - nếu L không đạt mà LN, LB hoặc LS đạt → nguyên nhân tương ứng (a), (b) hoặc (c).
3. Không biến thể nào đạt → bác bỏ cả bốn giả thuyết, báo cáo kết quả âm.
4. **Thay mô hình đã chọn** là quyết định riêng của người dùng, không tự động. Biến thể ứng viên phải:
   - có MAE trung bình 3 mức nhiễu camera (0,005 / 0,007 / 0,010) trên validation_ext không kém hetero quá 0,005 mm;
   - có độ phủ ±1,96σ trong khoảng 90–98 %.

**Lưu ý:** validation_ext đã được dùng để so sánh các biến thể ở bước 04 (biến thể), nên đây là lần dùng lặp. Con số cuối cùng cho bài báo vẫn chỉ đến từ locked_test.

## Kết quả (12/12 lượt huấn luyện xong, không lỗi, không NaN)

Ensemble 3 seed, 8 vị trí dịch. Đơn vị: mm. Nguồn: `exp2_layered/05_error_analysis/results/retraining/eval.json`.

| Mô hình | RMSE train σ=0 | RMSE val_ext σ=0 (chính) | RMS sai lệch val | MAE camera TB | Phủ ±1,96σ val σ=0 / 0,005 | corr(sai lệch val, derm_musp) |
|---|---|---|---|---|---|---|
| hetero (đã chọn) | 0,203 | **0,204** | 0,203 | 0,200 | 94 % / 94 % | 0,54 |
| L | 0,029 | 0,212 | 0,211 | 0,206 | 41 % / 69 % | 0,52 |
| LN | 0,030 | 0,213 | 0,213 | 0,212 | 48 % / 53 % | 0,51 |
| LB | 0,029 | 0,220 | 0,220 | 0,206 | 28 % / 63 % | 0,52 |
| LS | 0,024 | 0,210 | 0,209 | 0,206 | 44 % / 85 % | 0,53 |

### Áp tiêu chí

1. Không biến thể nào giảm RMSE chính ≥ 0,03 mm (tất cả **tăng** 0,006–0,016 mm). RMSE train giảm ~0,17–0,18 mm.
2. → Theo tiêu chí 3: **bác bỏ cả bốn giả thuyết** (huấn luyện ngắn/chọn checkpoint, (a), (b), (c)). Kết quả âm.
3. Không biến thể nào đủ điều kiện thay mô hình (MAE camera kém hơn 0,006–0,012 mm; độ phủ 28–85 %). hetero giữ nguyên.

### Điều mới học được (cần sửa cách hiểu của bước 05 (sai lệch cố định))

- bước 05 (sai lệch cố định) kết luận "học chưa tới" vì sai số train ≈ val. bước 05 (huấn luyện lại) cho thấy mạng **có thể** khớp train gần hoàn hảo (0,03 mm), nhưng sai số trên da chưa gặp không đổi (~0,21 mm). Vậy train ≈ val ở hetero là do dừng sớm, không phải do mạng thiếu khả năng.
- Sai lệch cố định là **sai số tổng quát hóa sang da mới**, và có cùng dạng ở mọi biến thể (corr với derm_musp 0,51–0,54). Nó không phụ thuộc cách huấn luyện.
- Còn hai cách giải thích, chưa phân biệt được:
  - (i) 300 da huấn luyện quá ít để học hàm từ profile sang độ sâu → sẽ giảm khi thêm da (kiểm bằng learning curve theo số da với công thức L, chỉ CPU);
  - (ii) cận bước 05 (giới hạn 14 tham số) (tuyến tính hóa) quá lạc quan: các da khác nhau cho profile gần giống nhau theo cách phi tuyến → đây mới là giới hạn thông tin thật.
- Mô hình dừng sớm (hetero) cho độ không chắc chắn được hiệu chỉnh tốt hơn hẳn; huấn luyện lâu làm σ quá tự tin trên da mới.
- Bằng chứng có sẵn nghiêng về (ii): đường cong học ở bước 04 (biến thể) (cách huấn luyện cũ) phẳng từ 50 đến 300 da (MAE 0,187 → 0,182). Nhưng đường cong đó dùng mạng dừng sớm, nên chưa loại hẳn (i). Phép kiểm dứt điểm: lặp lại đường cong học 50/100/200/300 da với công thức L.
