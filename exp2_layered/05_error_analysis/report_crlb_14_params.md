# Bước 05 — Giới hạn thông tin 14 tham số (CRLB) trên validation_ext

`locked_test` vẫn đóng. Các điểm tính được lấy từ validation_ext, là bộ đã dùng để so mô hình. Không bộ nào trong số này dùng để huấn luyện.

## Câu hỏi

Sai số của mô hình hetero/wide có một sàn khoảng 0,2 mm không phụ thuộc nhiễu (bước 04 (biến thể)). Ta đã loại ba nguyên nhân: thiếu dữ liệu, mạng nhìn quá hẹp, và nhiễu dữ liệu huấn luyện. Câu hỏi còn lại: sàn này có phải **giới hạn thông tin của phép đo** không, nghĩa là các tổ hợp lớp da khác nhau cho profile gần giống nhau?

## Vì sao CRLB cục bộ ở bước 02 chưa đủ

- CRLB cục bộ tỉ lệ thuận với σ nên tiến về 0 khi nhiễu tiến về 0. Nó không thể tạo ra một sàn.
- Một sàn không phụ thuộc nhiễu xuất hiện khi Fisher có **hướng gần suy biến** gắn với độ sâu. Khi đó sai số bị chặn bởi độ rộng phân bố tiên nghiệm, không bởi nhiễu.
- bước 02 chỉ có 9 tham số và chỉ tại một làn da chuẩn. bước 03 có 14 tham số:
  - lớp da: mel_fraction, dermis_mm, derm_musp, derm_mua, fat_mm, fat_musp, fat_mua, musc_musp, musc_mua;
  - mạch máu: depth, diameter, mua, mus;
  - vị trí: shift (dịch ngang).

## Cách tính

**Điểm tính**
- 40 mạch từ validation_ext, phân tầng theo độ sâu: 10 mạch mỗi dải [<2, 2–3, 3–4, ≥4] mm. Trong mỗi dải, chọn theo thứ tự manifest, mỗi làn da tối đa một mạch.
- Mạch phải còn hợp lệ sau mọi nhiễu loạn: z ≥ da + 0,3 mm và z + D ≤ da + fat − 0,9 mm.

**Jacobian**
- Sai phân trung tâm trên GPU (pmcxcl, cùng hình học và cách đọc như bước 03). Dùng chung số ngẫu nhiên (cùng seed) giữa bản + và bản −.
- Bước sai phân:
  - depth ±0,2 mm, diameter ±0,4 mm, fat_mm ±0,5 mm, dermis_mm ±0,2 mm;
  - mua và mus của máu ±10 % (theo log);
  - các tham số quang học lớp da ±20 % (theo log).
- Đạo hàm theo shift tính giải tích từ profile cơ sở.
- Mỗi profile chạy 2 nửa độc lập × 24 triệu photon.
- Profile = R / N0 vô hướng. Mỗi nhiễu loạn lớp da có N0 riêng (12 triệu photon). Profile được bin như đầu vào mạng: 128 bin × 0,25 mm.

**Fisher không chệch**
- F = ½ (J_aᵀ J_b + J_bᵀ J_a), với J_a và J_b tính từ hai nửa độc lập. Cách này loại phần phóng đại do nhiễu MC trong Jacobian.
- Đối chứng: F_naive = J̄ᵀ J̄.

**Tham số hóa và tiên nghiệm** (khớp cách lấy mẫu bước 03)
- Tham số log-uniform được biểu diễn bằng log. Phương sai tiên nghiệm Gaussian bằng phương sai của phân bố uniform tương ứng (range²/12).
- Riêng depth: dùng phương sai thực nghiệm của độ sâu trong validation_ext.

**Đầu ra tại mỗi σ ∈ {0,0035; 0,004; 0,005; 0,007; 0,010; 0,014}**
1. CRLB tần suất: (a) chỉ depth, (b) đủ 14 tham số. Cả hai tỉ lệ với σ.
2. **Cận Bayes tuyến tính hóa** cho depth: [(F/σ² + Σ_prior⁻¹)⁻¹]₀₀. RMSE của cận được tính là căn của trung bình qua 40 điểm (rủi ro Bayes của mô hình tuyến tính Gaussian).
3. Phân tích trị riêng: tổ hợp tham số nào làm depth gần suy biến.

**Mô hình đối chiếu**
- hetero_seed42/43/44 và wide (ensemble). Mô hình được chấm trên đúng 40 profile cơ sở này (48 triệu photon), có cộng nhiễu tới σ tổng, 64 bản mỗi điểm.
- Báo cáo RMSE, độ chệch và độ phân tán cục bộ.

## Tiêu chí kết luận (cố định từ bây giờ)

- **Sàn là giới hạn thông tin** khi cả hai điều sau cùng đúng:
  1. RMSE của cận Bayes trong khoảng 75–125 % RMSE của mạng ở mọi σ;
  2. cận Bayes cũng có sàn: tỉ lệ RMSE giữa σ = 0,014 và σ = 0,0035 nhỏ hơn 2,5, trong khi một cận tỉ lệ thuận với σ cho tỉ lệ 4.
- **Sàn do mô hình** (vẫn còn chỗ cải thiện) khi RMSE của cận Bayes < 75 % RMSE của mạng.
- **Ngoài hai trường hợp trên:** báo cáo là chưa kết luận được.
- Lưu ý: tuyến tính hóa trên toàn bề rộng tiên nghiệm chỉ là xấp xỉ. Ta sẽ báo cáo cả độ nhạy theo bước sai phân, bằng cách so hai nửa với trung bình.

## Chi phí

- Mỗi điểm: 27 cấu hình mạch × 2 × 24 triệu photon, cộng 19 N0 × 12 triệu photon, khoảng 1,5 tỉ photon, khoảng 27 phút.
- 40 điểm: khoảng 18 giờ GPU, một tiến trình GPU.
- Chạy lại được giữa chừng (mỗi điểm một file). Gặp kết quả invalid thì dừng.

## Sửa đổi phương pháp (sau điểm 1/40, trước khi có kết quả tổng thể)

**Vấn đề:** ở điểm pt_00, Jacobian của các tham số yếu bị nhiễu MC lấn át.
- Tương quan giữa hai nửa trước khi làm trơn: fat_mm −0,11; musc_musp 0,14; dermis_mm 0,27.
- F không chệch (J_aᵀJ_b) vì thế có trị riêng âm.
- Khi nghịch đảo, phần chéo của các tham số này gần 0 còn phần ngoài chéo thì nhiễu, tạo ra tương quan giả với depth. Cận Bayes ra 0,18 mm, lớn hơn cả RMSE thật của mạng tại điểm đó (0,066 mm), tức là vô lý.

**Sửa**
1. Chiếu mỗi cột Jacobian lên cơ sở B-spline bậc 3 (35 hàm, nút cách 1 mm). Phần dư khi chiếu chia cho nhiễu có trung vị 0,73–1,83 (≈ 1 nghĩa là phần bỏ đi gần như chỉ là nhiễu). Riêng cột shift là 1,83: phần bỏ đi có lẫn chút tín hiệu, nên thông tin theo shift bị đánh giá hơi thấp.
2. Ước lượng chính: **co rút James–Stein theo cột**, J_js = J̄·max(0, 1 − năng lượng nhiễu / năng lượng cột), rồi F = J_jsᵀJ_js.
3. Đối chứng lạc quan: F_naive = J̄ᵀJ̄, coi nhiễu là thông tin.
4. F không chệch vẫn được ghi lại, nhưng không dùng để kết luận.

**Tiêu chí kết luận** giữ nguyên, áp dụng cho cận JS.

**Lưu ý:** cận Bayes tuyến tính hóa không phải cận dưới chặt cho từng điểm riêng. Mạng có thể tốt hơn cận tại một điểm nhờ dùng cấu trúc của phân bố tiên nghiệm. Vì vậy chỉ so trên trung bình của cả 40 điểm.

## Kết quả (40/40 điểm)

Chạy `crlb_analysis.py` trên đủ 40 điểm; không có file `.invalid` hay NaN. Phần dư khi chiếu chia cho nhiễu có trung vị 0,85–1,16 ở mọi cột.

| σ | Cận JS (mm) | Cận naive | hetero RMSE | wide RMSE | JS / hetero |
|---|---|---|---|---|---|
| 0,0035 | 0,111 | 0,086 | 0,187 | 0,187 | 0,60 |
| 0,004 | 0,125 | 0,096 | 0,192 | 0,194 | 0,65 |
| 0,005 | 0,150 | 0,117 | 0,206 | 0,207 | 0,73 |
| 0,007 | 0,196 | 0,154 | 0,239 | 0,240 | 0,82 |
| 0,010 | 0,255 | 0,203 | 0,285 | 0,291 | 0,89 |
| 0,014 | 0,317 | 0,259 | 0,326 | 0,351 | 0,97 |

- Tỉ lệ RMSE giữa σ = 0,014 và σ = 0,0035: cận JS 2,84, naive 3,01; hetero 1,75, wide 1,87.
- Theo độ sâu, ở σ = 0,0035 (10 điểm mỗi dải):

  | Dải | Cận JS | hetero | Thành phần lệch của hetero |
  |---|---|---|---|
  | 1–2 mm | 0,064 | 0,195 | 0,192 |
  | 2–3 mm | 0,075 | 0,195 | 0,187 |
  | 3–4 mm | 0,100 | 0,214 | 0,192 |
  | 4–5 mm | 0,173 | 0,132 | 0,103 |

  Sai số của mạng tại mỗi điểm gần như hoàn toàn là độ lệch cố định cho cấu hình da đó, không phải tản do nhiễu. Ở dải 4–5 mm mạng tốt hơn cận, đúng như lưu ý về tiên nghiệm (mạng co về biên trên 5 mm).

**Áp tiêu chí ghi trước**
- Điều kiện "giới hạn thông tin" **không đạt**:
  - Tỉ lệ JS / mạng là 0,60–0,97, không nằm trong 75–125 % ở mọi σ.
  - Cận không có sàn: tỉ lệ 2,84 > 2,5.
- "Sàn do mô hình" (cận < 75 % mạng) **đạt ở σ = 0,0035; 0,004 và 0,005**, tức đúng vùng nhiễu thấp, nơi cái sàn xuất hiện.
- Ở σ ≥ 0,007, mạng đạt 82–97 % cận, tức gần tối ưu.
- Tiêu chí không ghi rõ trường hợp "< 75 %" có cần đúng ở mọi σ hay không. Ghi nhận diễn giải: sàn là hiện tượng ở nhiễu thấp, nên dùng các σ thấp.

**Kết luận:** cái sàn ~0,19 mm ở nhiễu thấp **không phải giới hạn thông tin của phép đo**. Ảnh chứa đủ thông tin cho khoảng 0,11 mm (0,06–0,10 mm với mạch < 4 mm). Phần thừa nằm ở mô hình, dưới dạng độ lệch cố định theo cấu hình da.

Hai phép thử trước đã loại nguyên nhân do nhiễu dữ liệu huấn luyện (bước 04 (biến thể) ghép cặp) và do vùng nhìn hẹp (wide ≈ hetero). Nguyên nhân cụ thể chưa xác định.

**Lưu ý**
- Cận tuyến tính hóa trên toàn bề rộng tiên nghiệm chỉ là xấp xỉ.
- Chỉ có 40 điểm.
- Cận naive (lạc quan) còn thấp hơn, nên không đổi kết luận.
