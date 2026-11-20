# Kế hoạch mô phỏng bổ sung cho luận văn và bài báo F/K

Đây là kế hoạch, chưa phải kết quả.

Kế hoạch này nối tiếp kiểm chứng mô phỏng V1–V6 đã hoàn tất (60/60 điều kiện đạt chất lượng MC, xem `exp1_homogeneous/06_final_test/results/report/KET_LUAN_V2_V6.md`). Nó bao gồm mọi phần mô phỏng còn cần cho hai mục tiêu:

1. Hoàn thành kiểm chứng phantom (V7–V8) theo [kế hoạch kiểm chứng độc lập](Ke_hoach_tiep_theo_kiem_chung_doc_lap_FK.md).
2. Bổ sung các phân tích mà reviewer tạp chí quốc tế tầm trung sẽ đòi: đối chứng không dùng CNN, mô hình camera, độ bền với các yếu tố ngoài hấp thụ.

Phần Raspberry Pi và chế tạo phantom không thuộc kế hoạch này.

## 1 Nguyên tắc giữ nguyên

| Nội dung | Quy định |
| --- | --- |
| File đã khóa | Không sửa `configs/simulation.json`, `configs/learning.json`, `configs/validation_independent.json`, `code/simulate.py`, `code/readout.py`, `code/train.py` và sáu checkpoint trong `exp1_homogeneous/04_training/results/FK/models`. Mọi thay đổi đi vào file cấu hình và module **mới**. |
| Công cụ | Giữ `pmcxcl==0.7.5` (OpenCL); không thay bằng `pmcx`. Không chạy hai bộ chạy GPU cùng lúc. |
| Bộ kiểm chứng mới | Mỗi bộ có manifest riêng, khóa bằng SHA-256 **trước** khi chạy MC; chọn dữ liệu theo chất lượng MC **trước** khi mở dự đoán; cùng tiêu chí hội tụ và bậc photon như V1. |
| Dữ liệu đã mở | Test giai đoạn 4 và bộ V1 đã được xem kết quả. Chỉ được báo trên đó như kết quả phụ, ghi rõ "đã mở"; không dùng để chọn phương pháp hay chỉnh mô hình. |
| Mô hình mới | Nếu cần mô hình phiên bản 2 (mục S5), nó phải có bộ kiểm chứng riêng và được khóa **trước** khi mở ảnh của năm khối phantom cuối. Báo cả sáu checkpoint cũ lẫn phiên bản 2. |
| Lỗi MC | `.invalid.npz` hoặc NaN: giữ lại, dừng, không sửa. Không loại điều kiện vì CNN sai. |

## 2 Tổng quan các phần

| Mã | Phần | Cần GPU | Phụ thuộc | Vai trò |
| --- | --- | --- | --- | --- |
| S0 | Hạ tầng và khóa phiên bản | Ít | — | Bắt buộc |
| S1 | Phương pháp đối chứng không dùng CNN | Không | S0 | Bắt buộc cho bài báo |
| S2 | Mô hình quan sát camera | Không | S0; tham số thật từ V7 | Bắt buộc cho phantom |
| S3 | Độ bền của sáu checkpoint với yếu tố ngoài hấp thụ | Có, ~7 giờ | S0, S1 khóa xong | Tăng giá trị bài báo |
| S4 | Mô phỏng khớp phantom và dự báo đăng ký trước | Có, ~2 giờ | S2, mẫu DEV của V7 | Bắt buộc cho phantom |
| S5 | Mô hình phiên bản 2 (có điều kiện) | Có, ~1 giờ MC + huấn luyện CPU | Quy tắc quyết định ở S2/S3 | Chỉ làm khi đạt điều kiện |
| S6 | Tổng hợp, hình và gói tái lập | Không | Các phần trên | Bắt buộc |

Thời gian GPU ước theo tốc độ đo trên RTX 3060: khoảng 66 s cho một lượt kênh 6 × 10⁷ photon và khoảng 130 s cho một N0 1,2 × 10⁸ photon. Tổng danh nghĩa khoảng 11 giờ GPU; trường hợp xấu nhất (mọi điều kiện phải tăng tới bậc photon cao nhất) gấp khoảng 7 lần.

## 3 Chi tiết từng phần

### S0 Hạ tầng và khóa phiên bản

| Việc | Cách làm | Điều kiện hoàn thành |
| --- | --- | --- |
| S0.1 Module mở rộng | Tạo `code/simulate_ext.py` gọi lại `simulate.make_volume` và `simulate.run`, chỉ thay hàm tạo bảng vật liệu để cho phép tán xạ kênh có hệ số g riêng. Không sửa `simulate.py`. | Với cấu hình mặc định: bảng vật liệu và `identity_hash` trùng hệt `simulate.py` (kiểm không cần GPU). Chạy lại một lượt V1 (ví dụ `mua_0.508_depth_2.300_rep1`) với cùng seed và so bằng tiêu chí hội tụ 10%, vì cộng dồn trên GPU có thể không tất định đến từng byte. |
| S0.2 Cấu hình mới | Thư mục `configs/sim2/`: một file cho mỗi biến thể (tán xạ nền, đường kính, tán xạ kênh). Mỗi file chỉ khác cấu hình gốc ở đúng tham số được khảo sát. | Kiểm tự động: so từng trường với `simulation.json`, chỉ cho phép khác ở trường đã khai báo. |
| S0.3 Thông số máu | Chốt hệ số tán xạ và g của máu toàn phần ở 850 nm từ tài liệu (ví dụ tổng quan về tính chất quang của máu của Bosschaart và cộng sự, 2014; cần kiểm lại nguồn trước khi dùng). Ghi nguồn, trang và điều kiện hematocrit. | Có bảng nguồn giống Phụ lục A của kế hoạch gốc. Không tự đặt số. |
| S0.4 Script khóa | Tổng quát hóa `v0_lock_manifest.py` thành bộ tạo manifest dùng chung cho S3, S4, S5: điều kiện, seed MC, photon, hash cấu hình, hash checkpoint và chỉ số định trước. | Unit test kiểm hash và kiểm không trùng seed với V1. |
| S0.5 Unit test | Thêm test cho các phần trên. Giữ 17 test gốc. | Toàn bộ test đạt. |

### S1 Phương pháp đối chứng không dùng CNN

Mục đích: cho thấy CNN thêm được gì so với cách đơn giản, và lợi ích của huấn luyện đa hấp thụ có xuất hiện cả ở phương pháp không học sâu hay không.

| Việc | Cách làm |
| --- | --- |
| S1.1 Đặc trưng | Từ biên dạng 128 điểm (đúng phép trích `v1`): tương phản C = 1 − giá trị nhỏ nhất, độ rộng FWHM, diện tích vùng tối. Quy tắc tính khóa trước, kể cả khi FWHM không xác định. |
| S1.2 B1: nghịch đảo một mức | Đường chuẩn C → độ sâu dựng từ bảng tra μa = 0,508 (dữ liệu K). Đây là phiên bản "không học sâu" của K. |
| S1.3 B2: hồi quy đa hấp thụ | Hồi quy trên (C, FWHM, diện tích) học từ tập train F, ví dụ đa thức bậc ≤ 3 hoặc k-NN. Chọn bậc hoặc k bằng **validation giai đoạn 4**. Đây là phiên bản "không học sâu" của F. |
| S1.4 Khóa | Lưu tham số B1/B2 và hash trước khi khóa manifest S3. |
| S1.5 Đánh giá | Kết quả chính trên S3 (bộ mới). Báo thêm trên test giai đoạn 4 và V1 với nhãn "đã mở". Cùng chỉ số với V3–V6. |

Không dùng thêm GPU. Không cần cài thêm thư viện ngoài NumPy/SciPy đã có.

### S2 Mô hình quan sát camera

Mục đích: chuyển bản đồ phản xạ lý tưởng thành ảnh giống camera thật, để đầu vào mô phỏng và phantom cùng đi qua một chuỗi xử lý.

| Bước | Nội dung | Nguồn tham số |
| --- | --- | --- |
| S2.1 Lấy mẫu pixel | Chuyển `R_xy` (voxel 0,1 mm) sang lưới pixel theo mm/pixel. | Đo trên thước chuẩn ở V7. |
| S2.2 Độ nhòe | Tích chập với PSF ước lượng từ ảnh cạnh sắc hoặc khe. | V7. |
| S2.3 Mức tín hiệu | Quy đổi sao cho N0 mô phỏng có cùng mức sáng trung bình với N0 thật ở phơi sáng đã khóa. | V7. |
| S2.4 Nhiễu | Mô hình Poisson–Gauss: phương sai = a·tín hiệu + b, theo phương pháp của Foi và cộng sự (2008). Áp riêng cho ảnh có kênh và ảnh N0; N0 lấy trung bình đúng số khung như quy trình thật. | Ước a, b từ 5 mức sáng × 20 ảnh ở V7. |
| S2.5 Lượng tử hóa và offset | Độ sâu bit và ảnh tối; trừ tối ở cả tử và mẫu trước khi chia N0. | V7. |
| S2.6 Trích biên dạng | Dùng lại phép trích 128 điểm trên 32 mm × 8 mm sau khi quy đổi theo mm/pixel. | Không đổi. |

Lưu thành phiên bản `observation/v2`, kèm hash tham số. Trước khi có số đo V7, viết và kiểm mã với một **khoảng tham số giả định** để biết sai số nhạy với nhiễu và độ nhòe đến đâu. Kết quả giai đoạn này chỉ là phân tích độ nhạy, không gán là đặc tính Camera Module 3 NoIR.

Kiểm tra: khi tắt nhiễu và độ nhòe, đầu ra phải trùng với phép trích `v1` trong sai số làm tròn.

### S3 Độ bền của sáu checkpoint với các yếu tố ngoài hấp thụ

Mục đích: định lượng mô hình đã khóa chịu được bao nhiêu khi thực tế khác giả định huấn luyện. Không huấn luyện lại; chỉ đánh giá.

**Thiết kế một yếu tố một lần**, mỗi biến thể chỉ khác cấu hình tham chiếu ở một tham số:

| Biến thể | Giá trị | Cần N0 mới |
| --- | --- | --- |
| R Tham chiếu | μs′ nền 1,0 mm⁻¹; D = 3 mm; kênh không tán xạ | Có (3 lượt, seed mới) |
| A1, A2 Tán xạ nền | μs′ = 0,7 và 1,3 mm⁻¹ | Có (3 lượt mỗi mức) |
| B1, B2 Đường kính | D = 2 và 4 mm | Không, dùng N0 của R |
| C Tán xạ kênh | Máu có tán xạ, thông số từ S0.3 | Không, dùng N0 của R |
| N Nhiễu camera | Áp `observation/v2` lên bản đồ của R | Không chạy MC |

**Lưới cho mỗi biến thể:** 5 độ sâu {0,8; 1,6; 2,6; 3,6; 4,4} mm (không trùng nút huấn luyện và độ sâu V1) × 4 mức μa {0,15; 0,22; 0,508; 1,0} × 3 lượt MC lặp = 60 lượt.

**Khối lượng:** 6 biến thể có MC × 60 = 360 lượt, khoảng 6,6 giờ, cộng 9 N0, khoảng 20 phút. Biến thể C có thể chậm hơn vì photon tán xạ nhiều trong kênh; đo thời gian ở pilot.

**Trình tự:** khóa manifest → pilot (z = 0,8 và 4,4 mm tại μa = 0,15 cho mỗi biến thể) → toàn bộ lưới → chọn dữ liệu theo chất lượng MC → dự đoán sáu CNN cùng B1/B2 → tổng hợp.

**Chỉ số định trước:**
- Với mỗi biến thể: MAE F, K, B1, B2 theo nhóm μa; ΔMAE = MAE_K − MAE_F; từng seed và trung bình ± SD.
- Mức suy giảm so với R: MAE(biến thể) − MAE(R) trên cùng độ sâu và μa.
- Biến thể nào làm F mất lợi thế so với K, và ở vùng độ sâu nào.

**Kết luận được phép:** mô tả độ bền của mô hình đã khóa trong khoảng tham số đã thử; không suy rộng ra mọi loại da. Nếu một biến thể làm hỏng mạnh, đó là giới hạn cần báo, không phải lý do để bỏ biến thể đó.

### S4 Mô phỏng khớp phantom và dự báo đăng ký trước

Mục đích: trước khi chụp năm khối cuối, ghi lại mô phỏng **dự báo** kết quả F/K trên phantom thế nào. So dự báo với kết quả thật cho thấy khoảng cách mô phỏng–thực tế.

| Việc | Cách làm | Khối lượng |
| --- | --- | --- |
| S4.1 Ước lượng μs′ hiệu dụng của nền | Dùng mẫu DEV có kênh ở độ sâu đã đo. So biên dạng tỉ số thật (sau `observation/v2`) với lưới mô phỏng μs′ ∈ {0,7; 0,85; 1,0; 1,15; 1,3} mm⁻¹ tại μa của dung dịch P1. Chọn μs′ có sai khác biên dạng nhỏ nhất theo tiêu chí định trước. Nếu có chuẩn phản xạ, dùng thêm phản xạ tuyệt đối. | 5 mức × 3 lượt = 15 lượt kênh + 15 N0, khoảng 50 phút |
| S4.2 Lưới khớp phantom | Năm độ sâu danh nghĩa 0,75 / 1,25 / 2 / 3 / 4,5 mm × các dung dịch phantom × 3 lượt lặp, với μs′ từ S4.1 và μa đo hoặc ước lượng của từng dung dịch. | Với 3 dung dịch: 45 lượt + 3 N0, khoảng 55 phút |
| S4.3 Dự báo | Áp `observation/v2`; cho sáu CNN và B1/B2 dự đoán; lưu MAE và ΔMAE dự kiến cho từng khối và dung dịch. | Không thêm GPU |
| S4.4 Đăng ký trước | Khóa hash của dự báo **trước** khi chụp năm khối cuối. | — |
| S4.5 Sau khi cắt mặt cắt | Nếu độ sâu đo thật lệch nhiều so với danh nghĩa, chạy lại mô phỏng ở độ sâu đo thật. Việc này hợp lệ vì MC không phụ thuộc kết quả CNN. Báo cả dự báo đã đăng ký và bản cập nhật. | Tối đa 45 lượt |

**Khuyến nghị:** thêm dung dịch P3 ở mức F chưa học (khoảng 0,22 mm⁻¹) vào phantom; lưới S4.2 đã tính sẵn cho 3 dung dịch.

**Giới hạn cần ghi:** mô phỏng dùng nguồn phẳng và biên ngang tuần hoàn, trong khi phantom dùng vòng LED và khối hữu hạn 100 × 80 × 20 mm. Phép chia N0 khử được một phần khác biệt chiếu sáng, không khử hết.

### S5 Mô hình phiên bản 2 (có điều kiện)

**Quy tắc quyết định, đặt trước:** chỉ huấn luyện phiên bản 2 nếu ít nhất một điều kiện sau xảy ra:
- Áp mô hình camera đo thật (`observation/v2`) lên bộ R của S3 làm MAE trung bình của F ở nhóm μa {0,15; 0,22} tăng hơn 0,10 mm so với quan sát lý tưởng.
- Mẫu DEV cho thấy biên dạng thật lệch khỏi mô phỏng vượt ngưỡng hội tụ 10% sau khi đã hiệu chỉnh μs′ ở S4.1.

Nếu không xảy ra, giữ sáu checkpoint cũ cho phantom và ghi rõ đã kiểm điều kiện.

| Việc | Cách làm |
| --- | --- |
| S5.1 Dữ liệu | Tạo lại bộ F/K từ bảng tra `exp1_homogeneous/03_dataset/data/lut_FK.npz` có sẵn, đi qua `observation/v2` với nhiễu ngẫu nhiên theo cảnh. Không cần MC mới cho dữ liệu huấn luyện. Nếu S4.1 cho μs′ khác 1,0 rõ rệt, cần bảng tra mới; khi đó lập kế hoạch riêng vì tốn khoảng 48 lượt × 4 μa. |
| S5.2 Cấu hình học | `configs/learning_v2.json`: giữ kiến trúc; tăng số epoch tối đa (ví dụ 300) và độ kiên nhẫn dừng sớm, vì F ở phiên bản 1 dừng ở epoch 100/100/97, đúng trần. Chọn checkpoint bằng validation. |
| S5.3 Bộ kiểm chứng riêng | 5 độ sâu mới {0,7; 1,2; 2,1; 3,1; 4,1} mm × 4 μa × 3 lượt = 60 lượt + 3 N0, khoảng 1,2 giờ. Khóa trước khi mở dự đoán. |
| S5.4 Khóa | Khóa sáu checkpoint phiên bản 2 trước khi mở ảnh năm khối cuối. |

### S6 Tổng hợp, hình và gói tái lập

| Sản phẩm | Nội dung |
| --- | --- |
| Bảng chính | F, K, B1, B2 trên V1 (đã có), S3, S4 dự báo và phantom; từng seed và trung bình ± SD. |
| Hình | Bản đồ sai số theo z × μa cho từng biến thể; dự đoán theo μa (như V4); suy giảm so với tham chiếu; dự báo S4 so với kết quả phantom. |
| Gói tái lập | Manifest, hash, cấu hình, phiên bản công cụ và lệnh chạy cho mọi bộ; hướng dẫn chạy lại. Dùng làm phụ lục luận văn và tài liệu công khai kèm bài báo. |

## 4 Thứ tự và lịch dự kiến

| Thứ tự | Việc | Ước thời gian | Ghi chú |
| --- | --- | --- | --- |
| 1 | S0 | 3–4 ngày | Có thể làm ngay |
| 2 | S1 | 3–4 ngày | Song song với chế tạo phantom |
| 3 | S2 mã và phân tích độ nhạy giả định | 1 tuần | Song song |
| 4 | Khóa S3 → chạy S3 | 1 đêm GPU + 1–2 ngày tổng hợp | Sau khi S1 khóa |
| 5 | V7: đo camera, mẫu DEV | Theo tiến độ phòng thí nghiệm | Cung cấp tham số cho S2 |
| 6 | S2 với tham số thật → S4.1 → S4.2–S4.4 | 3–4 ngày | Trước khi chụp năm khối cuối |
| 7 | Quyết định S5 → nếu cần, S5 | 1–2 tuần | Phải xong trước bước 8 |
| 8 | V8: chụp năm khối cuối, cắt mặt cắt, so với dự báo S4 | Theo tiến độ phòng thí nghiệm | |
| 9 | S6 | 1 tuần | |

## 5 Ngoài phạm vi luận văn

Các hướng sau dành cho giai đoạn sau (xem [kế hoạch mô phỏng hướng tới tạp chí hàng đầu](Ke_hoach_mo_phong_tap_chi_hang_dau_FK.md)), không đưa vào kế hoạch này: đo thêm ở cùng 850 nm với cấu hình chiếu sáng hoặc phân cực khác để tách độ sâu khỏi hấp thụ ở vùng sâu, mô hình da nhiều lớp có melanin, phân tích khả năng nhận diện bằng thông tin Fisher, ước lượng độ bất định của dự đoán, và thử nghiệm trên người với siêu âm tham chiếu.
