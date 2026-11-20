# Bước 06 — Chấm bộ test khóa (locked_test), mở đúng một lần

Kế hoạch ghi **trước khi mở** `locked_test`.
Script: `exp2_layered/06_final_test/code/score_locked_test.py`. Kết quả: `exp2_layered/06_final_test/results/locked_test/`.

## Dữ liệu

- `exp2_layered/03_dataset/data/locked_test`: 40 cấu hình da × 3 tĩnh mạch = 120 tĩnh mạch, 60 triệu photon mỗi kênh, seed 20261115 (bước 03).
- 40 file `.npz`, không có file `.invalid`. Chưa bước nào trước đây đọc nội dung các file này.
- Đọc theo đúng readout bước 03: profile = `profiles_y_mean / n0_scalar`; kiểm `config_json` trùng với manifest.

## Mô hình cuối (đã chọn ở bước 04 (biến thể), khóa bằng SHA-256)

Mô hình cuối là **hetero**, ensemble 3 seed, huấn luyện trên 300 làn da train (12 triệu photon).
- Lý do chọn (tiêu chí bước 04 (biến thể)): hetero 0,1955 và wide 0,1954 mm chênh nhau dưới 0,01 mm, nên ưu tiên hetero vì có độ bất định.
- Mục 4 của kế hoạch cải thiện ở bước 04 (huấn luyện lại trên train + train_ext) **không làm**, vì train_ext đã hoãn.
- bước 05 (huấn luyện lại) không thay mô hình: không biến thể nào đạt tiêu chí.

| File | SHA-256 |
|---|---|
| t5b_variants/hetero_seed42.pt | b0c908e98173e78b2b7240410001a9754fa7ce5e5d85277a21b30dd5a1d2b5e7 |
| t5b_variants/hetero_seed43.pt | eaa7b263a6b6397224c53de36b33e8b0d2105c863945428746aae10e23e69e7c |
| t5b_variants/hetero_seed44.pt | d0f3086e88015f6a6d3e790baadd2433d5187043620ea310950c2a28490afcd9 |

Mô hình so sánh (chấm trên cùng bộ, theo mục 5 của kế hoạch cải thiện ở bước 04):

| Mô hình | File | SHA-256 |
|---|---|---|
| wide (về nhì ở bước 04 (biến thể)) | t5b_variants/wide_seed42/43/44.pt | 92da4cfd… / 012a6949… / 2ec1049d… |
| F_layered (bước 04) | t5_layered/models/F_layered_seed42/43/44.pt | e0db2873… / e3f3571b… / 97c0ae49… |
| F đã khóa (đồng nhất, thử nghiệm 1) | stage4/models/cnn_F_seed42/43/44.pt | 15f97faa… / 1a69c8c5… / 52cfa983… |
| B3 đồng nhất (khớp vật lý) | stage4/lut_FK.npz | 71e7ec15… |
| Hằng số = trung bình độ sâu train | — | — |

Hash đầy đủ được script ghi lại vào `exp2_layered/06_final_test/results/locked_test/hashes.json` và so với bảng trên trước khi đọc dữ liệu. Nếu lệch, script dừng.

## Cách chấm (giống `compare_validation_ext.py` trên validation_ext)

- σ_mc đo từ chính locked_test, ở vùng xa (|x| > 12 mm, gộp bin).
- Nhiễu camera thêm vào = √(σ_tổng² − σ_mc²), với σ_tổng ∈ {0,005; 0,007; 0,010}.
- Thêm một mức phụ: chỉ có nhiễu MC (không thêm nhiễu).
- 4 bản dịch ngang U(−2, 2) mm cho mỗi mạch; seed bộ sinh nhiễu 900 + k.
- B3: chỉ chấm 1 bản dịch mỗi mạch, như ở bước 04.

## Chỉ số

**Chỉ số chính:** MAE của ensemble hetero, trung bình ba mức nhiễu chính.

Báo cáo thêm:
- MAE, RMSE, bias theo từng mức nhiễu;
- MAE theo dải độ sâu (< 2, 2–3, 3–4, ≥ 4 mm);
- MAE theo melanin (< 0,05 / ≥ 0,15);
- độ phủ ±1σ / ±1,645σ / ±1,96σ của hetero;
- khoảng tin cậy 95 % bootstrap theo **làn da** (40 cấu hình, 2000 lần lấy mẫu);
- cùng các chỉ số cho mô hình so sánh.

## Cách đọc kết quả (cố định)

1. Con số cuối cho bài báo là chỉ số chính ở trên, kèm khoảng tin cậy. Báo cáo **dù tốt hay xấu**.
2. **Nhất quán với validation_ext** (0,1955 mm cùng cách tính) khi |chênh lệch| ≤ 0,03 mm. Nếu lệch hơn, báo cáo và tìm nguyên nhân; **không** chỉnh mô hình theo locked_test.
3. Độ bất định hetero được coi là hiệu chuẩn tốt khi độ phủ ±1,96σ nằm trong 90–98 %.
4. Sau khi mở, mọi cải tiến mô hình phải được chấm trên một bộ test khóa **mới**.

## Ràng buộc

- Mở đúng một lần. Script tạo `exp2_layered/06_final_test/results/locked_test/OPENED.json` trước khi đọc dữ liệu và từ chối chạy nếu file này đã tồn tại.
- Không huấn luyện, không chọn hay chỉnh gì theo locked_test.

## Trước khi mở

- Đã chạy thử toàn bộ script trên validation_ext (`--dry-run-on-validation-ext`, kết quả ở `exp2_layered/06_final_test/results/dryrun_validation_ext/`): không lỗi.
  - Chỉ số chính hetero 0,193 mm (khớp 0,1955 ở bước 04 (biến thể); chênh do bộ sinh nhiễu khác).
  - Không tạo cờ OPENED.
- SHA-256 của `exp2_layered/06_final_test/code/score_locked_test.py` lúc mở: `cea7b2af794d47974d9f898ede972fc8eb99de043028c6ec22af56a333ee4841`.

## Kết quả (mở một lần)

`exp2_layered/06_final_test/results/locked_test/eval.json`. 120 tĩnh mạch trên 40 làn da; σ_mc của bộ này là 0,0023.

### Chỉ số chính: kết quả cuối cho bài báo

**MAE của hetero (ensemble 3 seed), trung bình ba mức nhiễu = 0,206 mm** (KTC 95 % bootstrap theo làn da: 0,179–0,235 mm).
- validation_ext cho 0,1955 mm. Chênh +0,010 mm, nhỏ hơn 0,03 mm → **nhất quán** (tiêu chí 2).
- Độ phủ ±1,96σ là 92,5 / 94,2 / 95,6 % → nằm trong 90–98 %, **độ bất định hiệu chuẩn tốt** (tiêu chí 3).

### Theo mức nhiễu (hetero)

| Nhiễu tổng mỗi bin | MAE | KTC 95 % | RMSE | Bias | Phủ ±1,96σ | σ dự báo trung vị |
|---|---|---|---|---|---|---|
| chỉ MC (0,0023) | 0,175 | 0,140–0,215 | 0,231 | +0,088 | 90,4 % | 0,18 |
| 0,005 | 0,185 | 0,154–0,219 | 0,242 | +0,047 | 92,5 % | 0,20 |
| 0,007 | 0,202 | 0,176–0,232 | 0,260 | +0,036 | 94,2 % | 0,23 |
| 0,010 | 0,231 | 0,201–0,265 | 0,295 | +0,018 | 95,6 % | 0,26 |

MAE theo dải độ sâu ở σ = 0,005: < 2 mm 0,14; 2–3 mm 0,21; 3–4 mm 0,19; ≥ 4 mm 0,23. Ở σ = 0,010, mạch ≥ 4 mm có MAE 0,34.
MAE theo melanin ở σ = 0,005: da sáng (< 0,05) 0,16; da sẫm (≥ 0,15) 0,26.

### So sánh (trung bình ba mức nhiễu, cùng bộ test)

| Mô hình | MAE (mm) |
|---|---|
| **hetero (mô hình cuối)** | **0,206** |
| wide | 0,207 |
| F_layered (bước 04) | 0,224 |
| F đã khóa (huấn luyện trên da đồng nhất) | 1,232 |
| B3 khớp vật lý đồng nhất | 1,270 (chạm cận 5 mm ở 25–27 % số mạch) |
| Hằng số = độ sâu trung bình train | 0,859 |

### Đọc kết quả

- Trên 40 làn da chưa từng dùng ở bất kỳ bước nào, mô hình cuối sai trung bình khoảng 0,2 mm, bằng 24 % sai số của cách đoán hằng số. Kết quả khớp với validation_ext.
- Mô hình huấn luyện trên da đồng nhất (F đã khóa) và phép khớp vật lý đồng nhất (B3) sai khoảng 1,2 mm trên da nhiều lớp. Nghĩa là mô hình da phải nhiều lớp thì mới dùng được.
- Độ bất định mạng tự báo khớp với sai số thật ở các mức nhiễu camera.
- Hạn chế còn lại vẫn như bước 05: da sẫm và mạch sâu sai nhiều hơn; có bias dương nhỏ (đoán sâu hơn thật 0,02–0,05 mm).
- locked_test đã mở. Từ nay mọi cải tiến mô hình cần một bộ test khóa mới.
