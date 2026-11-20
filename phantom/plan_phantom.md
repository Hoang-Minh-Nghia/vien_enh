# Phantom — Kiểm chứng trên phantom vật lý (kế hoạch, chưa làm)

Chưa đúc, chưa chụp phantom nào. Mọi con số dưới đây là tính toán thiết kế, chạy bằng `code/design_budget.py`, kết quả trong `results/design/budget.json`. Script chỉ đọc kết quả có sẵn (Jacobian bước 05 (giới hạn 14 tham số) và dự đoán đã lưu của bước 06), không chạy MC, không chấm lại locked_test.

## 1. Phantom phải trả lời câu hỏi gì

Ảnh tay thật (phần ảnh tay thật) không có độ sâu chuẩn. locked_test (bước 06) cho 0,206 mm, nhưng hoàn toàn trên mô phỏng. Phantom lấp khoảng giữa hai bước đó. Có ba câu hỏi, **tách riêng** nhau:

| | Câu hỏi | Cách trả lời |
|---|---|---|
| H1 | Mô phỏng MC có tả đúng ảnh thật không? | profile phantom đo được so với MC của chính phantom đó (với μa, μs′, độ dày đã đo) |
| H2 | Mạng đoán độ sâu trên ảnh thật sai bao nhiêu? | độ sâu mạng đoán so với độ sâu đo hình học |
| H3 | Mạng có nhầm "hấp thụ yếu" với "nằm sâu" không? (lỗi gân/mạch mà luận văn Đàm 2025 gặp) | hai kênh cùng độ sâu, μa máu giả khác nhau |

Thêm H4, đi theo bước 05 (sai lệch cố định): sai lệch cố định của mạng theo cấu hình da có xuất hiện đúng chiều trên phantom không (so sai lệch đo được của từng khối với sai lệch MC dự báo cho khối đó).

Bài so sánh (Đàm và cs., IEEE Access 13:195704, 2025, doi:10.1109/ACCESS.2025.3633201) chỉ làm H2. Ở đó có một khối ức gà, độ sâu chuẩn tính từ góc nghiêng, và chỉ báo hệ số tương quan 0,86. Họ không đo tính chất quang của khối, nên không tách được lỗi mô phỏng với lỗi mạng.

## 2. Ba phép tính thiết kế

### 2.1 Cần đo tính chất quang chính xác tới đâu (chỉ quan trọng cho H1 và B3)

Câu hỏi: nếu đo sai một tham số của phantom 1 sd, rồi dùng MC với giá trị sai đó để giải thích profile bằng độ sâu, thì độ sâu lệch bao nhiêu? Cách tính: chiếu cột Jacobian bước 05 (giới hạn 14 tham số) của tham số lên cột độ sâu (và dịch ngang). Bảng dưới là trung vị qua 40 điểm validation_ext.

| Tham số (1 sd) | Mức "thường gặp" | Lệch (mm) | Mức "chặt" | Lệch (mm) |
|---|---|---|---|---|
| μa lớp chứa mạch ("mỡ") | ±20 % | **0,15** (≥ 4 mm: 0,22) | ±5 % | 0,04 |
| μs′ lớp "bì" | ±10 % | **0,09** | ±5 % | 0,05 |
| μs′ lớp "mỡ" | ±10 % | 0,06 (≥ 4 mm: 0,11) | ±5 % | 0,03 |
| μa máu giả | ±10 % | 0,04 | ±5 % | 0,02 |
| Độ dày "bì" | ±0,1 mm | 0,04 | ±0,05 mm | 0,02 |
| Đường kính kênh | ±0,1 mm | 0,03 | ±0,05 mm | 0,01 |
| μa "bì", μs máu, độ dày "mỡ", lớp "cơ" | | ≤ 0,03 mỗi cái | | ≤ 0,02 |
| **Tổng (cộng bình phương)** | | **0,23** | | **0,10** |

Đọc bảng:
- Ở mức đo "thường gặp", sai số do đo phantom (0,23 mm) **bằng cỡ sai số của mạng** (0,21 mm). H1 khi đó không kết luận được gì. Phải đạt mức "chặt".
- Hai tham số quyết định là **μa của lớp chứa mạch** và **μs′ của lớp bì**. Hai tham số này phải lấy theo công thức pha đã hiệu chuẩn, không nên chỉ đo trên khối.
- μa "mỡ" ≈ 0,01 mm⁻¹ là rất thấp. Riêng nước đã hấp thụ khoảng 0,004 mm⁻¹ ở 850 nm (Hale & Querry 1973, doi:10.1364/AO.12.000555), nên phải tính nước vào μa.
- Bảng này **không** áp dụng cho H2: mạng không biết tính chất quang của phantom. Với H2, chỉ cần phantom nằm trong dải huấn luyện (mục 3).

### 2.2 Độ sâu chuẩn cần chính xác tới đâu (cho H2)

Nếu độ sâu chuẩn có sai số ngẫu nhiên sd s, MAE quan sát được sẽ tăng (giả sử RMSE thật 0,26 và MAE 0,21 mm như bước 06):

| s (mm) | 0 | 0,05 | 0,10 | 0,15 | 0,20 | 0,30 |
|---|---|---|---|---|---|---|
| MAE quan sát | 0,210 | 0,214 | 0,225 | 0,242 | 0,265 | 0,321 |

Mục tiêu: **s ≤ 0,1 mm** (MAE tăng ≤ 7 %). Thước kẹp đơn thuần (±0,1–0,2 mm, cộng agar co ngót) chưa đủ. Phải dùng khuôn định vị và cắt kiểm (mục 4).

### 2.3 Cần bao nhiêu khối phantom

- Trên locked_test, sai số của mạng chủ yếu là **sai lệch cố định theo từng làn da**. sd của sai lệch theo làn da là 0,18 mm, lớn hơn MAE trung bình (0,18 mm).
- Một khối phantom tương đương **một làn da**, nên mọi điểm dọc một kênh chỉ là **một** mẫu độc lập. Đơn vị thống kê là khối, không phải điểm.

Nửa độ rộng KTC 95 % của MAE trung bình theo số khối (bootstrap từ 40 làn da của bước 06):

| Số khối | 1 | 2 | 3 | 4 | 6 | 8 | 10 |
|---|---|---|---|---|---|---|---|
| ± (mm) | 0,18 | 0,14 | 0,12 | 0,10 | **0,08** | 0,07 | 0,06 |

Hệ quả:
- Một khối như bài so sánh chỉ cho biết MAE nằm đâu đó trong 0–0,4 mm.
- **Tối thiểu 6 khối nhiều lớp**, mỗi khối một công thức da khác nhau.

## 3. Thiết kế khối

**Vật liệu:**
- agar 1,5 %, Intralipid-20 % (gây tán xạ), mực tàu (gây hấp thụ);
- công thức nền: Cubeddu và cs. 1997, doi:10.1088/0031-9155/42/10/011;
- giá trị tham chiếu μs′ của Intralipid-20 % và μa của mực tàu, với độ bất định khoảng 1–2 %: Spinelli và cs. 2014, doi:10.1364/BOE.5.002037.

Lưu ý: agar làm μs′ giảm một cách hệ thống (Cubeddu 1997), nên phải hiệu chuẩn **trong agar**, không dùng số đo trong nước. Tiêu chí thiết kế chung theo Hacker và cs. 2022, doi:10.1038/s41551-022-00890-6.

**Khối nhiều lớp (B1–B6), trong dải huấn luyện bước 03:**

| Lớp | Độ dày | μs′ (mm⁻¹) | μa (mm⁻¹) |
|---|---|---|---|
| "Bì" | 1,0 mm (B5: 0,7; B6: 1,4) | B1–B2: 1,6; B3–B4: 2,3; B5–B6: 2,9 | 0,02 |
| "Mỡ" (chứa kênh) | 8 mm | xen kẽ 1,1 / 1,9 | 0,008 / 0,015 |
| "Cơ" | ≥ 15 mm | 0,8 | 0,03 |

Các lựa chọn trong bảng:
- Dải μs′ bì được phủ có chủ ý, vì bước 05 (sai lệch cố định) chỉ ra sai lệch cố định chủ yếu đến từ tán xạ lớp bì.
- **Không có lớp biểu bì.** Như vậy tương đương melanin ≈ 1 %, mép dưới dải huấn luyện. Độ dày quang học của biểu bì 60 µm với M = 1 % chỉ ≈ 0,01. Da sẫm để đợt sau, dùng màng mỏng có mực.
- **Kênh mạch:**
  - cách đúc: thanh thép Ø 3,00 mm đặt trong khuôn in 3D, nghiêng 5,7° (≈ 0,1 mm độ sâu mỗi mm chiều dài); rút thanh ra sau khi gel đông, để lại kênh không có thành ống;
  - vị trí: mép trên chạy từ 1,2 mm xuống 5,0 mm trên 38 mm;
  - máu giả bơm vào kênh: mực + Intralipid; một kênh μa 0,30, một kênh μa 0,60 (hai đầu dải huấn luyện); μs′ ≈ 1,3.
- Hai kênh nghiêng **cùng chiều** trong mỗi khối, cách nhau ≥ 35 mm, để ở mỗi độ sâu đều có cặp μa thấp/cao (H3).
- Khối rộng ≥ 100 mm, dài ≥ 60 mm, để profile ±16 mm và vùng lấy nền không chạm mép.

**Khối đồng nhất (A1–A2):**
- một lớp, μs′ 1,0 và 1,8, μa 0,01, dày ≥ 30 mm;
- dùng cho H1, F_locked và B3, và để so trực tiếp với cách làm của bài báo;
- không dùng để chấm mô hình cuối (nằm ngoài dải huấn luyện).

Khác biệt còn lại so với mô phỏng, cần đưa vào MC của phantom:
- chiết suất agar ≈ 1,335 so với 1,37 trong mô phỏng;
- máu giả có g khác máu thật;
- kênh không có thành ống.

## 4. Độ sâu chuẩn (mục tiêu s ≤ 0,1 mm)

1. **Khuôn:** in 3D có rãnh định vị thanh và miếng chêm cho từng lớp. Đo độ dày từng lớp ở 4 góc bằng đồng hồ so hoặc thước đo sâu.
2. **Trước khi chụp:** chụp ngang qua thành khuôn trong suốt có thước, để lấy góc và điểm đầu của thanh.
3. **Sau khi chụp (chuẩn cuối):** cắt khối tại 6 vạch dọc mỗi kênh. Chụp macro mặt cắt có thước, đo mép trên kênh (độ phân giải ≈ 0,05 mm). Khớp một đường thẳng qua 6 điểm để có độ sâu tại mọi vị trí.
4. Nếu có siêu âm tần số cao thì đo thêm trước khi cắt. Việc này cũng là bước tập cho đo siêu âm trên tay về sau.
5. Chụp và cắt trong ngày đúc, vì agar mất nước làm đổi độ dày.

## 5. Đo tính chất quang (mục tiêu: mức "chặt")

- **μa (mực):** đo phổ truyền qua của dung dịch mực trong suốt bằng máy quang phổ, rồi pha loãng theo khối lượng.
- **μs′ (Intralipid trong agar 1,5 %):** đúc một loạt mẫu chuẩn ở 3 nồng độ. Chiếu chấm laser 850 nm, chụp quầng phản xạ theo bán kính bằng chính camera của hệ, khớp bằng MC (pmcxcl) hoặc mô hình khuếch tán (Farrell 1992, doi:10.1118/1.596777). Kiểm tính tuyến tính theo nồng độ (Cubeddu 1997).
- Mỗi khối đúc thêm một mẫu nhỏ cho mỗi lớp, cùng mẻ, để đo lại.
- Ghi độ bất định của từng giá trị và đưa vào phép tính ngân sách 2.1 cho đúng khối đó.

## 6. Chụp

- Cùng hệ chụp tay phần ảnh tay thật: camera, đèn 850 nm, khoảng cách, phơi sáng.
- Đặt thước trong khung; chụp dark frame.
- Mỗi khối chụp 5 lần, có dịch hoặc xoay khối giữa các lần.
- Ghi vùng chói do mặt agar bóng và loại vùng đó trước khi xử lý.
- Xử lý bằng pipeline phần ảnh tay thật **không sửa gì**. Lấy profile tại các điểm cách nhau 1 mm dọc kênh, bỏ 3 mm ở mỗi đầu kênh.

## 7. Phân tích và tiêu chí (cố định trước khi chụp)

**Mô hình:** hetero (mô hình cuối, khóa theo hash ở bước 06), F_layered, F_locked, B3 (với μa, μs′ đo được), và đoán hằng số. Không huấn luyện lại hay chỉnh gì theo dữ liệu phantom.

| | Chỉ số | Đạt khi |
|---|---|---|
| H1 | độ sâu khớp MC của phantom (B3 với tính chất đo được) so với độ sâu chuẩn | \|sai lệch trung bình\| ≤ 2 × ngân sách 2.1 của khối đó |
| H2 | MAE của hetero, trung bình theo **khối** (B1–B6), KTC bootstrap theo khối | MAE ≤ 0,30 mm và cận trên KTC ≤ 0,40 mm |
| H2 | độ phủ ±1,96σ của hetero | 85–98 % (thấp hơn bước 06 vì các điểm trong một khối tương quan) |
| H3 | chênh độ sâu đoán giữa kênh μa 0,60 và 0,30 ở cùng độ sâu | \|chênh\| ≤ 0,2 mm; ngưỡng so với MC của phantom sẽ ghi trước khi chụp |
| H4 | sai lệch đo được của từng khối so với sai lệch MC dự báo | cùng dấu ở ≥ 5/6 khối |

Báo cáo thêm:
- MAE, bias, độ dốc của đường đoán–thật theo dải độ sâu;
- hệ số tương quan, để đối chiếu với 0,86 của bài so sánh.

**Cách đọc kết quả:**
- H1 đạt, H2 đạt: mô phỏng và mạng đều dùng được trên vật thật.
- H1 đạt, H2 trượt: lỗi nằm ở mạng (khả năng tổng quát hóa).
- H1 trượt: mô phỏng chưa tả đúng hệ chụp (ví dụ nguồn sáng, chiết suất, camera). Sửa mô phỏng trước, **trên dữ liệu khối A**. Khi đó B1–B6 phải được đúc mới, vì đã bị nhìn.

## 8. Thứ tự làm và chi phí

1. Hiệu chuẩn mực và Intralipid trong agar: 1–2 buổi.
2. Khối A1–A2 và kiểm H1 sơ bộ. Đây là bước "chạy thử" giống dry-run ở bước 06, được phép chỉnh quy trình, nhưng không chỉnh mô hình.
3. Ghi bản cuối của mục 7 kèm hash script, rồi đúc và chụp B1–B6.
4. MC của từng khối với tính chất đo được: mỗi khối khoảng 1 giờ GPU, một tiến trình mỗi lúc.
5. Chấm một lần và báo cáo dù tốt hay xấu.

Không có người tham gia nên không cần duyệt đạo đức. Nếu dùng máu động vật thì cần ghi rõ nguồn.
