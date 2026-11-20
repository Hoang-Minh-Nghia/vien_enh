# Bước 04 — Kế hoạch cải thiện F_layered (các biến thể)

Ghi **trước** khi có kết quả so sánh. `locked_test` (40 cấu hình, 60 triệu photon) vẫn đóng.

## Vì sao làm

- F_layered (bước 04) đạt MAE khoảng 0,18 mm trên validation, gấp khoảng 2 lần giới hạn CRLB (đã quy đổi).
- Có hai nghi vấn về nguyên nhân:
  1. **Thiếu dữ liệu:** mới có 300 làn da độc lập.
  2. **Mạng nhìn quá hẹp:** DepthCNN chỉ có 3 lớp tích chập kernel 5, nên mỗi điểm chỉ "nhìn" được 13 bin (3,25 mm). Trong khi đó, vùng tối của mạch sâu rộng 10–20 mm.

## GPU (chạy liên tục) – sinh thêm dữ liệu, `exp2_layered/03_dataset/code/dataset_extend_ext.py`

- Manifest riêng `exp2_layered/03_dataset/data/manifest_ext.json`, seed mới, ghi trước khi chạy, không trùng id với bộ cũ.
- Vật lý, khoảng lấy mẫu và cách đọc giữ nguyên như bước 03.

| Tập | Cấu hình × mạch | Photon / mạch | Seed | Thời gian ước tính |
| --- | --- | --- | --- | --- |
| validation_ext | 90 × 5 | 24 triệu | 20261117 | ≈ 3,5 giờ (chạy trước) |
| train_ext | 900 × 5 | 12 triệu | 20261116 | ≈ 17 giờ |

## CPU (song song) – biến thể mô hình, `exp2_layered/common/cnn_variants.py`

- Tăng cường ngay khi huấn luyện: mỗi epoch dùng 8 bản mới cho mỗi mạch, với dịch ngang và nhiễu camera mới.
- Checkpoint được chọn trên validation cũ (30 da).
- Ba seed: 42, 43, 44.

| Biến thể | Nội dung |
| --- | --- |
| base | DepthCNN, LR cố định (đối chứng) |
| sched | DepthCNN, LR giảm khi chững (ReduceLROnPlateau) |
| wide | CNN giãn (dilated) nhìn được 125 bin ≈ 31 mm, có BatchNorm |
| multi | wide + mục tiêu phụ: đường kính, log μa máu, log melanin |
| hetero | wide + đầu ra (μ, σ), hàm mất mát Gaussian NLL → độ bất định (giả thuyết H3) |

## Tiêu chí chọn mô hình cuối (cố định từ bây giờ)

1. So mọi biến thể trên **validation_ext** (90 da mới, không dùng để chọn checkpoint), ở nhiễu tổng mỗi bin 0,005 / 0,007 / 0,010.
2. Chỉ số chính: **MAE trung bình ba mức nhiễu của ensemble 3 seed**.
3. Chọn biến thể có chỉ số chính nhỏ nhất. Nếu hai biến thể chênh nhau dưới 0,01 mm, chọn biến thể đơn giản hơn. Riêng hetero được ưu tiên nếu ngang ngửa, vì có thêm độ bất định cho H3.
4. Huấn luyện lại biến thể đã chọn trên train + train_ext (1200 da). Validation cũ chỉ dùng để chọn checkpoint, như trước.
5. Khóa mô hình bằng cách ghi hash của checkpoint, rồi mở **locked_test một lần** ở cùng ba mức nhiễu tổng. Báo cáo cả F đã khóa, F_layered (bước 04) và B3 trên cùng bộ test.

## Thay đổi kế hoạch (ghi trước phép so trên validation_ext)

**Bằng chứng 1 – đường cong học** (mạng wide, validation cũ, 3 seed, `exp2_layered/04_training/results/variants/wide_n*_seed*.json`):

| Số làn da | 50 | 100 | 200 | 300 |
| --- | --- | --- | --- | --- |
| MAE (mm) | 0,187 | 0,181 | 0,181 | 0,182 |

Từ 100 làn da trở lên, sai số không giảm nữa. Vậy số làn da không phải nút thắt.

**Bằng chứng 2 – phân tách sai số theo nhiễu** (`exp2_layered/04_training/code/noise_floor.py`, `exp2_layered/04_training/results/variants/noise_floor_validation.json`):
- Khớp RMSE² = sàn² + (k·σ)².
- Mạng wide: sàn ≈ 0,19 mm (sâu < 3 mm: 0,12 mm; ≥ 3 mm: 0,26 mm); k ≈ 15–20 mm trên mỗi đơn vị σ.
- Với mạch nông hơn 3 mm, k ≈ 15 trùng với CRLB cục bộ ở bước 02 (0,045 mm / 0,003). Nghĩa là trước nhiễu, mô hình đã dùng thông tin gần như tối ưu; phần thừa là một sàn không phụ thuộc nhiễu.

**Quyết định**
1. **Hoãn train_ext** (900 làn da × 12 triệu photon). Dừng tiến trình GPU ngay khi validation_ext đủ 90 làn da. Manifest_ext giữ nguyên để chạy sau nếu cần.
2. **Thay bằng bước 03 (dữ liệu nhiều photon)** (`exp2_layered/03_dataset/code/dataset_high_snr_a.py`, `manifest_hisnr.json`, seed 20261118): chạy lại 120 làn da train đầu tiên, giữ nguyên lớp da và mạch, với 96 triệu photon mỗi mạch (σ_mc ≈ 0,0017), khoảng 18 giờ GPU.
   - Giả thuyết: cái sàn do dữ liệu huấn luyện quá nhiễu.
   - Phép thử ghép cặp: cùng 120 làn da, huấn luyện ở 12 triệu photon so với 96 triệu photon. Chấm trên validation_ext ở các mức nhiễu tổng 0,004 / 0,005 / 0,007 / 0,010.
3. Tiêu chí chọn mô hình cuối (mục trên) giữ nguyên. Nếu bước 03 (dữ liệu nhiều photon) xác nhận giả thuyết, mô hình cuối được huấn luyện trên dữ liệu ít nhiễu.

## Kết quả so trên validation_ext (`exp2_layered/04_training/results/variants/compare_validation_ext.json`)

- 90 làn da × 5 mạch = 450 mạch; σ_mc đo được 0,00355 mỗi bin. Chưa mở locked_test.
- Chỉ số chính là MAE của ensemble 3 seed, trung bình qua ba mức nhiễu tổng 0,005 / 0,007 / 0,010.

| Mô hình | Chỉ số chính (mm) | Ở 0,005 | Ở 0,010 |
| --- | --- | --- | --- |
| wide | 0,1954 | 0,172 | 0,220 |
| hetero | 0,1955 | 0,171 | 0,220 |
| multi | 0,1973 | 0,174 | 0,221 |
| F_layered (bước 04) | 0,2081 | 0,186 | 0,233 |
| sched | 0,2086 | 0,189 | 0,232 |
| base | 0,2203 | 0,200 | 0,244 |
| F_locked (luận văn) | 1,153 | 1,149 (lệch +1,08) | 1,159 |

- Độ bất định của hetero được hiệu chuẩn tốt. Độ phủ thực tế ở mức nhiễu 0,005: 64 % / 88 % / 94 %, so với danh nghĩa 68 % / 90 % / 95 %. Hệ số Spearman giữa σ dự đoán và sai số tuyệt đối là 0,28–0,37.
- **Áp dụng tiêu chí đã cố định:** wide và hetero chênh nhau dưới 0,01 mm, nên **chọn hetero**. Huấn luyện cuối sẽ dùng 12 triệu hay 96 triệu photon tùy kết quả bước 03 (dữ liệu nhiều photon).

## bước 03 (dữ liệu nhiều photon) – phép thử ghép cặp 12 triệu và 96 triệu photon (ghi trước khi huấn luyện)

**Kiểm tra dữ liệu bước 03 (dữ liệu nhiều photon)** (120 làn da × 5 mạch)
- Không có NaN, không có file invalid.
- σ_mc trung vị: 0,00487 ở 12 triệu photon, 0,00178 ở 96 triệu (tỉ lệ 2,73; lý thuyết √8 = 2,83).
- Vùng xa đã chuẩn hóa: 0,9989 ± 0,0009.
- Độ tương phản (1 − min) khớp cặp với bản 12 triệu: r = 0,998.

**Thiết kế**
- Hai nhóm, khác nhau duy nhất ở số photon của dữ liệu huấn luyện:
  - **A (12 triệu):** `--nskin=120`, tức 120 làn da đầu của train.
  - **B (96 triệu):** `--hisnr`, đúng 120 làn da đó ở 96 triệu photon.
- Mọi thứ khác giữ nguyên:
  - tăng cường dữ liệu: dịch ngang U(−2, 2) mm, nhiễu cộng log-uniform [0,001; 0,01];
  - checkpoint chọn trên validation cũ, như trước;
  - siêu tham số không đổi.
- Biến thể: hetero (đã chọn) và wide; seed 42, 43, 44.

**Chấm điểm**
- Trên validation_ext (σ_mc 0,00355), ở nhiễu tổng 0,004 / 0,005 / 0,007 / 0,010.
- Báo cáo ensemble MAE, RMSE, phân tách sàn/k, và MAE theo dải độ sâu.

**Tiêu chí kết luận**
- Giả thuyết "sàn do dữ liệu huấn luyện nhiễu" **được ủng hộ** khi cả hai điều sau cùng đúng:
  1. MAE ensemble của B thấp hơn A ít nhất 0,01 mm, tính trung bình bốn mức nhiễu, ở cả hetero lẫn wide;
  2. sàn ước lượng của B thấp hơn của A.
- Ngược lại, kết luận là sàn **không** do nhiễu dữ liệu huấn luyện.

**bước 03 (dữ liệu nhiều photon) – GPU, chạy song song với phép thử trên CPU** (`exp2_layered/03_dataset/code/dataset_high_snr_b.py`, `manifest_hisnr2.json`, seed 20261119)
- Chạy lại 180 làn da train còn lại (thứ tự 120–299) ở 96 triệu photon, khoảng 27 giờ GPU. Khi xong sẽ có đủ 300 làn da 96 triệu photon.
- Chỉ dùng cho huấn luyện cuối, và chỉ khi phép thử ghép cặp ủng hộ giả thuyết. Bộ này không ảnh hưởng gì đến phép thử ghép cặp.

### Kết quả phép thử ghép cặp (`exp2_layered/04_training/results/paired/paired_eval.json`)

**MAE ensemble (mm) theo nhiễu tổng mỗi bin trên validation_ext.** A = huấn luyện ở 12 triệu photon, B = ở 96 triệu.

| Nhiễu tổng | hetero A | hetero B | wide A | wide B |
| --- | --- | --- | --- | --- |
| 0,0035 (chỉ MC) | 0,169 | 0,158 | 0,172 | 0,160 |
| 0,004 | 0,170 | 0,162 | 0,174 | 0,164 |
| 0,005 | 0,181 | 0,173 | 0,181 | 0,173 |
| 0,007 | 0,193 | 0,187 | 0,196 | 0,189 |
| 0,010 | 0,213 | 0,211 | 0,223 | 0,214 |
| 0,014 | 0,249 | 0,242 | 0,256 | 0,244 |
| **Trung bình bốn mức chính** | **0,189** | **0,183** | **0,193** | **0,185** |

**Sàn ước lượng (RMSE, mm)**
- hetero: 0,208 → 0,197.
- wide: 0,212 → 0,201.
- k không đổi, khoảng 18,6.

**Kết luận theo tiêu chí đã ghi trước**
- B tốt hơn A một cách nhất quán, ở mọi mức nhiễu, mọi seed và cả hai biến thể.
- Mức giảm là 0,006 mm (hetero) và 0,008 mm (wide), **dưới ngưỡng 0,01 mm**.
- Sàn chỉ giảm khoảng 5 %.
- → **Giả thuyết "sàn do dữ liệu huấn luyện nhiễu" bị bác bỏ.** Nhiễu dữ liệu huấn luyện chỉ đóng góp nhỏ, khoảng 0,01 mm của sàn. Sàn khoảng 0,2 mm (≈ 0,16 mm với mạch nông hơn 3 mm, ≈ 0,25 mm với mạch sâu từ 3 mm) đến từ nguồn khác.
- Ứng viên chính: bài toán không định danh được hoàn toàn, khi các tổ hợp lớp da khác nhau cho profile gần giống nhau. Có thể kiểm bằng CRLB đa tham số, có tính các tham số gây nhiễu (nuisance) của lớp da.
