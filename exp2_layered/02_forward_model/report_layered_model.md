# Bước 02 — Mô hình da nhiều lớp ở 850 nm

**Mô hình** (thông số và nguồn ở `T0_bang_tham_so.md`, mục "Bổ sung cho bước 02"):

- biểu bì 0,06 mm, bì 1,1 mm, mỡ 6 mm, cơ;
- tĩnh mạch Ø 3 mm chứa máu toàn phần **có tán xạ**;
- độ sâu tính từ mặt da đến mép trên tĩnh mạch: 1,6 / 2,6 / 3,6 mm;
- da sáng (melanosome 3 %) và da sẫm hơn (15 %).

Mỗi kênh chạy 60 triệu photon, mỗi N0 chạy 120 triệu photon; tổng 66 lượt, không có lượt lỗi. Mã nguồn: `exp2_layered/02_forward_model/code/layered_runs.py`, `exp2_layered/02_forward_model/code/layered_analysis.py`, `exp2_layered/02_forward_model/code/split_mismatch.py`, `exp2_layered/02_forward_model/code/split_fit.py`.

## 1. Độ tương phản

| Độ sâu (mm) | Nền đồng nhất, mạch có tán xạ | Da nhiều lớp, da sáng | Da nhiều lớp, da sẫm |
| --- | --- | --- | --- |
| 1,6 | 0,278 | 0,153 | 0,141 |
| 2,6 | 0,161 | 0,084 | 0,078 |
| 3,6 | 0,100 | 0,050 | 0,047 |

- Lớp da làm độ tương phản giảm khoảng một nửa.
- Melanin chỉ làm giảm thêm 6–8 %, vì phép chuẩn hóa theo N0 khử phần lớn ảnh hưởng của lớp hấp thụ đồng đều phía trên.
- Tán xạ trong máu tự nó làm độ tương phản giảm 15–20 % so với kênh không tán xạ cùng μa.

## 2. Độ chính xác giới hạn (CRLB) với 9 tham số

Chín tham số: độ sâu; ln μa của máu; dịch ngang; đường kính; ln μa biểu bì (melanin); ln μs′ bì; ln μs′ mỡ; ln μa mỡ; độ dày bì.

| Độ sâu (mm) | Hệ số phạt khi thêm đủ 9 tham số | √CRLB ở σ = 0,001 | ở σ = 0,003 | ở σ = 0,01 |
| --- | --- | --- | --- | --- |
| 1,6 | 5,2 | 0,014 mm | 0,043 mm | 0,145 mm |
| 2,6 | 3,2 | 0,014 mm | 0,043 mm | 0,144 mm |
| 3,6 | 2,0 | 0,015 mm | 0,046 mm | 0,154 mm |

- Như ở bước 02, μa của máu là tham số làm hại nhiều nhất: hệ số phạt 1,9–4,1.
- Nếu xét riêng từng tham số lớp da thì mỗi cái có hại: hệ số phạt 1,0–3,1, lớn nhất là μs′ bì và melanin.
- Nhưng khi μa máu đã nằm trong mô hình, việc thêm năm tham số lớp da gần như không làm tăng giới hạn nữa: gain khi bỏ từng tham số chỉ 1,00–1,07. Các tham số này tác động lên profile theo cùng hướng với μa máu.

**Kết luận:** nếu mô hình thuận đúng, thông tin về độ sâu vẫn đủ. Giới hạn ≈ 0,045 mm ở σ = 0,003, cùng cỡ với mô hình đồng nhất ở bước 02.

## 3. Sai lệch mô hình — kết quả quan trọng nhất

Dùng B3 (khớp vật lý theo mô hình đồng nhất, kênh không tán xạ, tức cùng giả định với mô hình F/K đã huấn luyện của luận văn) cho dữ liệu da nhiều lớp:

| Độ sâu thật (mm) | Chỉ có máu tán xạ (nền đồng nhất) | Da nhiều lớp, da sáng | Da nhiều lớp, da sẫm |
| --- | --- | --- | --- |
| 1,6 | +0,34 mm (dư 0,0029) | +0,97 mm (dư 0,0073) | +1,19 mm (dư 0,0083) |
| 2,6 | +0,33 mm (dư 0,0029) | +1,18 mm (dư 0,0053) | +1,42 mm (dư 0,0061) |
| 3,6 | +0,30 mm (dư 0,0031) | +1,39 mm (dư 0,0045) | +1,40 mm (dư 0,0055) |

"Dư" là sai số dư RMS mỗi bin.

- Mô hình đồng nhất **luôn đoán sâu hơn thật**, lệch 1,0–1,4 mm. Sai số này lớn hơn mười lần sai số trong mô phỏng.
- Trên dữ liệu da nhiều lớp, μa ước lượng chạm cận dưới 0,15. Nghĩa là mô hình không còn giải thích được dữ liệu.
- Khoảng 0,3 mm đến từ tán xạ trong máu. Phần này khó phát hiện vì sai số dư 0,003 chỉ gần mức nhiễu MC.
- Phần còn lại, khoảng 0,7–1,1 mm, đến từ lớp da. Sai số dư lúc này gấp 2–3 lần nhiễu, nên có thể dùng để phát hiện mô hình sai.
- Cách tách hai phần trên chỉ là xấp xỉ: lượt nền đồng nhất dùng g = 0,511 và n = 1,35, còn lượt da nhiều lớp dùng g = 0,9 và n = 1,37.

## Hàm ý

1. **Với luận văn:** kết quả hiện tại vẫn đúng cho phạm vi đã khai báo, tức môi trường đồng nhất với kênh không tán xạ, đúng loại phantom bạn sẽ làm. Phải ghi rõ trong phần giới hạn rằng trên da thật, mô hình này có thể lệch khoảng 1 mm, có số liệu từ bước 02 kèm theo.
2. **Với bài báo:** mô hình thuận dùng để sinh dữ liệu huấn luyện phải là **da nhiều lớp với máu tán xạ**, với các thông số lớp da được lấy ngẫu nhiên trong khoảng bảng tham số. Đây là bước 03. Thông tin đủ (theo CRLB), nhưng chỉ khi mô hình đúng.
3. **Với phantom:** phantom đồng nhất kiểm được phần cứng và quy trình, nhưng không kiểm được tuyên bố "trên da". Nên thêm một phantom có lớp phủ tán xạ mỏng (khoảng 1 mm, μs′ khoảng 2,3 mm⁻¹) để kiểm đúng hiệu ứng ở mục 3.

## Giới hạn

- Biểu bì được dồn vào một voxel 0,1 mm, có hiệu chỉnh độ dày quang học.
- Ba thông số là ước lượng: μa bì, μa mỡ, độ dày mỡ.
- CRLB là cận cục bộ với nhiễu trắng.
- Chỉ có ba độ sâu và một đường kính tĩnh mạch.
- Tất cả đều là mô phỏng; chưa có dữ liệu đo.
