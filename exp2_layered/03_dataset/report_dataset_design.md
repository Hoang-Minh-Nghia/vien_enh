# Bước 03 — Bộ dữ liệu da nhiều lớp ở 850 nm (thiết kế)

Mã: `exp2_layered/03_dataset/code/dataset_generate.py`. Manifest: `exp2_layered/03_dataset/data/manifest.json`, viết **trước** mọi lượt chạy, từ seed cố định. Kiểm nhanh: `exp2_layered/03_dataset/code/dataset_check.py` (không bao giờ mở `locked_test`).

## Cấu trúc

| Tập | Số cấu hình da | Tĩnh mạch / cấu hình | Photon mỗi kênh | Photon N0 | Seed |
| --- | --- | --- | --- | --- | --- |
| train | 300 | 5 | 12 triệu | 6 triệu | 20261113 |
| validation | 30 | 5 | 12 triệu | 6 triệu | 20261114 |
| locked_test | 40 | 3 | 60 triệu | 12 triệu | 20261115 |

Tổng: 1770 tĩnh mạch, khoảng 8,5 giờ GPU.

- Tập test có cấu hình da riêng và seed riêng, chạy cuối cùng. Chỉ được mở một lần, khi chấm mô hình cuối, theo cùng quy tắc với V1.
- Validation tách theo **cấu hình da**, không tách theo tĩnh mạch. Nhờ vậy mô hình được chấm trên những làn da nó chưa thấy.

## Cách đọc (readout t4)

- Mỗi cấu hình có một N0 không có tĩnh mạch.
- Vì lớp da đồng đều theo phương ngang và nguồn chiếu phẳng trên miền tuần hoàn, N0 là hằng số. N0 được rút thành **một con số**: trung bình phản xạ trên toàn bề mặt.
- Profile = trung bình theo y của phản xạ có tĩnh mạch, chia cho N0 vô hướng. Lưu 600 điểm, bước 0,1 mm.
- Dịch ngang, gộp về 128 bin và nhiễu camera được thêm lúc tạo dữ liệu huấn luyện, giống giai đoạn 4.
- Khác readout v1 của luận văn: v1 chia từng pixel cho một N0 có nhiễu. Kiểm trên pilot cho thấy tỉ số ở vùng xa tĩnh mạch đúng bằng 1,000 ± 0,001.

## Khoảng lấy mẫu (nguồn trong bảng tham số)

| Tham số | Khoảng | Phân bố | Căn cứ |
| --- | --- | --- | --- |
| Tỉ lệ melanosome biểu bì | 1–30 % | log-đều | Jacques 1998: 1,3–43 %; cắt ở 30 % |
| Độ dày bì | 0,6–1,5 mm | đều | Van Mulder 2016: da 0,65–1,55 mm |
| μs′ bì | 1,5–3,0 mm⁻¹ | đều | Bảng 1 của Jacques 2013 |
| μa bì | 0,01–0,04 mm⁻¹ | log-đều | ước lượng |
| Độ dày mỡ | 4–10 mm | đều | **giả định** |
| μs′ mỡ | 1,0–2,1 mm⁻¹ | đều | Simpson 1998; Salomatina 2006 |
| μa mỡ | 0,005–0,02 mm⁻¹ | log-đều | ước lượng quanh Simpson 1998 |
| μs′ / μa cơ | 0,6–1,0 / 0,02–0,04 mm⁻¹ | đều | Jacques 2013; ước lượng |
| Độ sâu tĩnh mạch (mép trên) | (da + 0,1) – 5,0 mm | đều | Mukai 2020; Ichimura 2020 |
| Đường kính tĩnh mạch | 2–4 mm | đều | Sharp 2018 |
| μa máu | 0,30–0,60 mm⁻¹ | log-đều | Bosschaart 2014, nới rộng theo Hct/SO₂ |
| μs máu | 60–85 mm⁻¹ (g = 0,983) | đều | Bosschaart 2014 |

Ràng buộc: tĩnh mạch phải nằm trọn trong lớp mỡ. Lớp mô dùng g = 0,9 và n = 1,37. Biểu bì 60 µm được dồn vào một voxel 0,1 mm, nhân hệ số 0,6 để giữ độ dày quang học.

## Bước sau (bước 04)

1. Tạo dữ liệu huấn luyện gồm dịch ngang, nhiễu σ ∈ [0,001; 0,01] và 128 bin.
2. Huấn luyện F_layered với cùng kiến trúc DepthCNN, tối đa 400 epoch, chọn theo validation.
3. So sánh với:
   - F đã khóa, huấn luyện trên mô hình đồng nhất;
   - B3 đồng nhất;
   - một phép khớp vật lý dùng mô hình da nhiều lớp, nếu đủ thời gian.
4. Mở `locked_test` đúng một lần, ở cuối.
