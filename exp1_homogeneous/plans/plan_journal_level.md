# Kế hoạch mô phỏng hướng tới tạp chí hàng đầu

Đây là kế hoạch, chưa phải kết quả.

Kế hoạch này chỉ gồm phần **mô phỏng và mô hình hóa** của lộ trình tạp chí hàng đầu (ví dụ JBO, Biomedical Optics Express, IEEE TMI). Phantom có đo tính chất quang và dữ liệu người được lập kế hoạch riêng. Kế hoạch cho luận văn ở [Ke_hoach_mo_phong_bo_sung_FK.md](Ke_hoach_mo_phong_bo_sung_FK.md) là giai đoạn nền; các phần S1 (đối chứng đặc trưng), S2 (mô hình camera) và S3 (độ bền) được dùng lại ở đây.

**Ràng buộc phần cứng: chỉ có một bước sóng 850 nm.** Kế hoạch không dùng đa bước sóng. Thông tin bổ sung để tách độ sâu khỏi hấp thụ phải đến từ **cấu hình đo ở cùng 850 nm**: vị trí và góc chiếu sáng, chiếu sáng có cấu trúc, và phân cực. Các phương án này dùng thêm linh kiện rẻ (bật tắt từng nhóm LED, tấm phân cực), không cần nguồn sáng mới.

Các số liệu vật lý cụ thể (độ dày lớp da, hệ số melanin, tán xạ của máu...) **không được tự đặt**: chốt từ tài liệu ở giai đoạn bảng tham số, ghi nguồn, trang và điều kiện. Các khả năng của công cụ được nêu dưới đây (lưu photon thoát, quãng đường từng môi trường, nguồn dạng mẫu) phải được kiểm trên `pmcxcl 0.7.5` ở pilot trước khi dựa vào.

## 1 Câu hỏi khoa học và giả thuyết đăng ký trước

Kết quả V1–V6 cho thấy: huấn luyện đa hấp thụ giảm mạnh sai số, nhưng vùng **mạch sâu + hấp thụ yếu** vẫn sai 0,6–0,7 mm. Câu hỏi của giai đoạn mới:

> Giới hạn đó là giới hạn **thông tin** của một ảnh phản xạ 850 nm với chiếu sáng đồng đều, hay giới hạn của **mô hình học**? Nếu là giới hạn thông tin, việc đo thêm ở cùng 850 nm với cấu hình chiếu sáng hoặc phân cực khác, kết hợp ước lượng đồng thời độ sâu và hấp thụ, có vượt qua được không, trong mô hình da thực tế?

Cơ sở vật lý: ở cùng bước sóng, photon thu được ở xa điểm chiếu sáng, hoặc photon đã mất phân cực, có xu hướng đi sâu hơn photon thu gần điểm chiếu hoặc còn giữ phân cực. Thay đổi cấu hình đo vì vậy thay đổi độ sâu mà ảnh nhạy nhất, có thể cho thông tin độc lập với μa. Mức thông tin thật sự phải được định lượng ở bước 02, không được giả định.

| Mã | Giả thuyết | Phép kiểm |
| --- | --- | --- |
| H1 | Với μa chưa biết, giới hạn Cramér–Rao (CRLB) của độ sâu từ một ảnh 850 nm chiếu sáng đồng đều tăng mạnh ở vùng sâu + hấp thụ yếu, và vùng CRLB lớn trùng vùng F sai nhiều. | bước 02 |
| H2 | Thêm ảnh ở cấu hình chiếu sáng hoặc phân cực khác (cùng 850 nm) làm giảm CRLB của độ sâu ở vùng đó; sai số thực của mô hình học giảm tương ứng. | bước 02, bước 04 |
| H3 | Mô hình có ước lượng độ bất định cho khoảng tin cậy hiệu chuẩn đúng (ví dụ khoảng 90% chứa giá trị thật khoảng 90% số lần), và báo độ bất định cao đúng ở vùng khó. | bước 04 |
| H4 | Lợi ích của đa dạng hóa tham số khi huấn luyện vẫn còn trong mô hình da nhiều lớp có tán xạ của máu. | bước 04 |

Kết luận được viết theo kết quả, kể cả khi giả thuyết không được hỗ trợ.

## 2 Tổng quan các giai đoạn

| Mã | Giai đoạn | Thời gian ước | Phụ thuộc | Cổng quyết định |
| --- | --- | --- | --- | --- |
| bảng tham số | Tổng quan tài liệu và bảng tham số | 1–2 tháng | — | Có bảng tham số có nguồn; xác định 1–2 phương pháp đã công bố để tái hiện |
| bước 01 | Mô hình thuận phiên bản 3 và kiểm chứng | 2 tháng | bảng tham số | Đạt các phép kiểm với lời giải giải tích và mã tham chiếu |
| bước 02 | Phân tích khả năng nhận diện | 1–2 tháng | bước 01 | H1 được hỗ trợ hay không; chọn cấu hình đo ở 850 nm |
| bước 02 | Chiến lược sinh dữ liệu hiệu quả | 1 tháng | bước 01 | Chi phí sinh dữ liệu trong ngân sách GPU |
| bước 03 | Sinh dữ liệu huấn luyện và bộ kiểm chứng khóa | 1–2 tháng | bước 02, bước 02 | Manifest khóa trước khi huấn luyện |
| bước 04 | Mô hình, đối chứng và ablation | 2 tháng | bước 03 | Đánh giá trên bộ khóa |
| bước 05 | Phân tích độ nhạy toàn cục và tổng hợp | 1 tháng | bước 04 | — |

Tổng khoảng **10–12 tháng** cho riêng phần mô phỏng, có thể chạy song song với phantom và chuẩn bị nghiên cứu trên người.

## 3 Chi tiết từng giai đoạn

### bảng tham số Tổng quan tài liệu và bảng tham số

| Việc | Nội dung |
| --- | --- |
| bảng tham số.1 Định vị | Tìm các công trình ước lượng độ sâu mạch từ ảnh phản xạ NIR, các công trình dùng Monte Carlo + học sâu cho bài toán ngược trong mô, và các công trình dùng nhiều cấu hình chiếu sáng hoặc phân cực ở một bước sóng. Ghi nhận cả công trình đa bước sóng để giải thích vì sao đề tài chọn hướng một bước sóng. Ghi rõ điểm mới của đề tài so với từng nhóm. |
| bảng tham số.2 Phương pháp để tái hiện | Chọn 1–2 phương pháp đã công bố có đủ mô tả để cài lại; ghi lý do chọn. |
| bảng tham số.3 Bảng tham số da | Độ dày và tính chất quang ở 850 nm của biểu bì (melanin), bì (thể tích máu, nước), mỡ dưới da, kèm khoảng dao động giữa các người. Nguồn gợi ý: tổng quan về tính chất quang của mô của Jacques (2013); cần kiểm lại trước khi dùng. |
| bảng tham số.4 Máu | μa ở 850 nm theo độ bão hòa oxy và hemoglobin từ phổ hemoglobin (Prahl/OMLC, như kế hoạch gốc); μs và g của máu toàn phần theo hematocrit từ tài liệu tổng quan về quang học máu. |
| bảng tham số.5 Khoảng biến thiên | Cho mỗi tham số: khoảng sinh lý dùng để đa dạng hóa, khoảng dùng làm kiểm ngoại suy, và nguồn. |
| bảng tham số.6 Cấu hình đo ứng viên ở 850 nm | Tìm tài liệu về phản xạ phân giải theo khoảng cách nguồn–thu, chiếu sáng có cấu trúc (miền tần số không gian) và ảnh phân cực song song/vuông góc cho bài toán độ sâu. Liệt kê các cấu hình làm được với vòng LED, camera NoIR và tấm phân cực. Chọn cuối cùng ở bước 02, không ở đây. |

### bước 01 Mô hình thuận phiên bản 3 và kiểm chứng

Tạo dự án con mới (ví dụ thư mục `forward_v3/`) để không lẫn với gói luận văn đã khóa.

| Việc | Nội dung | Điểm kỹ thuật cần kiểm |
| --- | --- | --- |
| bước 01.1 Da nhiều lớp | Biểu bì / bì / mỡ dưới da; mạch nằm trong bì hoặc mỡ. | Biểu bì mỏng cỡ voxel 0,1 mm hiện tại, cần voxel mịn hơn (0,025–0,05 mm) gần bề mặt hoặc dùng Monte Carlo trên lưới (MMC). Ở voxel 0,025 mm, miền 60 × 8 × 20 mm có khoảng 600 triệu voxel: phải tắt lưu fluence thể tích và kiểm bộ nhớ GPU 12 GB. |
| bước 01.2 Mạch máu | Máu có tán xạ (hàm pha Henyey–Greenstein, g cao); μa ở 850 nm theo độ bão hòa oxy; đường kính 1–5 mm. | Bước nhỏ trong kênh tán xạ mạnh làm chậm mô phỏng; đo thời gian ở pilot. |
| bước 01.3 Hình học mạch | Góc lệch so với trục y, độ cong nhẹ. | Bỏ biên tuần hoàn theo y khi mạch nghiêng; cần miền dài hơn theo y. |
| bước 01.4 Thu ảnh | Thay phản xạ thoát mọi hướng (`dref`) bằng photon thoát nằm trong góc nhận của camera đặt cách 200 mm. | Cần lưu vị trí và hướng thoát của photon; kiểm dung lượng lưu trữ. |
| bước 01.5 Chiếu sáng | Vòng LED với góc chiếu thật thay cho nguồn phẳng. Thêm các cấu hình ứng viên: bật từng nhóm LED (chiếu lệch một phía kênh), chiếu đường hoặc điểm ở các khoảng cách khác nhau so với kênh, chiếu sáng có cấu trúc theo tần số không gian. | Nguồn dạng mẫu hoặc nhiều nguồn; kiểm tính tuyến tính khi cộng nguồn. Nếu tính tuyến tính đúng, mô phỏng một lần cho từng nguồn cơ sở rồi cộng ra mọi cấu hình, giảm mạnh chi phí. |
| bước 01.6 Camera | Dùng lại mô hình camera S2 của kế hoạch luận văn. | — |
| bước 01.7 Phân cực | Ảnh phân cực song song và vuông góc (tấm phân cực trên nguồn và trên camera). Hiệu hai ảnh nhấn photon nông; ảnh vuông góc nhấn photon đi sâu. | Cần Monte Carlo có theo dõi phân cực. Phải kiểm `pmcxcl 0.7.5` có hỗ trợ không. Nếu không, cần công cụ khác đã được kiểm chứng; đây là quyết định riêng vì kế hoạch hiện tại chỉ dùng `pmcxcl`. |

**Kiểm chứng mô hình thuận (bắt buộc trước bước 02):**
- Nền đồng nhất, nguồn điểm: phản xạ theo bán kính so với lời giải khuếch tán của Farrell và cộng sự (1992) trong miền lý thuyết khuếch tán còn đúng.
- Nhiều lớp, không có mạch: so với MCML (Wang và cộng sự, 1995) hoặc một mã tham chiếu đã công bố.
- Hội tụ lưới, miền và photon theo cùng tiêu chí 10% của dự án, kiểm ở cả lớp mỏng nhất.
- Khi tắt các phần mở rộng, mô hình phiên bản 3 phải tái tạo kết quả phiên bản hiện tại trong sai số hội tụ.

### bước 02 Phân tích khả năng nhận diện

Đây là đóng góp phương pháp trung tâm của bài báo.

| Việc | Nội dung |
| --- | --- |
| bước 02.1 Tham số | Vector θ gồm độ sâu z, μa máu (hoặc độ bão hòa oxy), μs′ nền, đường kính D, melanin. Phân biệt tham số cần ước lượng (z) với tham số phiền nhiễu (phần còn lại). |
| bước 02.2 Jacobian | Đạo hàm của biên dạng theo từng tham số bằng sai phân hữu hạn trên MC. Với tham số hấp thụ, giữ cùng seed: MCX hạ trọng số photon khi hấp thụ, nên thay μa với cùng seed cho kết quả tương quan mạnh và sai phân ít nhiễu. Với tham số hình học (z, D), dùng bước sai phân đủ lớn và nhiều photon, kiểm độ ổn định khi đổi bước. |
| bước 02.3 Nhiễu | Ma trận hiệp phương sai nhiễu từ mô hình camera, cộng nhiễu MC còn lại. |
| bước 02.4 Thông tin Fisher và CRLB | Tính ma trận Fisher, CRLB của z khi (a) biết μa, (b) không biết μa, (c) không biết μa và μs′. Vẽ bản đồ CRLB theo (z, μa). |
| bước 02.5 Đối chiếu với kết quả thực nghiệm số | So bản đồ CRLB một bước sóng với bản đồ sai số của F trên V1 (đã có). Kiểm H1. |
| bước 02.6 Chọn cấu hình đo | Tính CRLB cho: (a) một ảnh chiếu đồng đều như hiện tại, (b) thêm ảnh chiếu lệch hoặc theo khoảng cách, (c) thêm cặp ảnh phân cực, (d) chiếu sáng có cấu trúc, và các tổ hợp. Chọn theo mức giảm CRLB ở vùng khó **và** tính khả thi phần cứng ở 850 nm. Khóa lựa chọn trước bước 03. |
| bước 02.7 Thông tin tiên nghiệm | Tính thêm trường hợp có thông tin tiên nghiệm về μa (khoảng sinh lý của hemoglobin và độ bão hòa oxy), dạng Bayes. Định lượng mức thông tin tiên nghiệm cần thiết để đạt mục tiêu sai số khi chỉ có một ảnh. |

Cổng:
- Nếu H1 không được hỗ trợ (CRLB không lớn ở vùng F thất bại), giới hạn nằm ở mô hình học chứ không ở phép đo. bước 04 chuyển trọng tâm sang kiến trúc và dữ liệu.
- Nếu H1 được hỗ trợ nhưng không cấu hình nào ở 850 nm giảm được CRLB đủ, đóng góp chính của bài trở thành: **xác định giới hạn lý thuyết của phương pháp một bước sóng, và mô hình có ước lượng độ bất định báo đúng khi nào không đo được**. Đây vẫn là kết quả có giá trị khoa học.
- Ghi quyết định và lý do trong cả hai trường hợp.

### bước 02 Chiến lược sinh dữ liệu hiệu quả

Không gian tham số nhiều chiều, nên cách hiện tại (bảng tra theo độ sâu rồi nội suy) không còn dùng được. Chạy MC trực tiếp cho từng mẫu cũng quá đắt: ở tốc độ hiện tại khoảng 66 s một lượt, 50.000 mẫu × 3 cấu hình chiếu sáng mất khoảng 115 ngày GPU.

| Phương án | Ý tưởng | Việc cần kiểm ở pilot |
| --- | --- | --- |
| P1 Hiệu chỉnh hấp thụ sau mô phỏng | Mỗi lượt MC chỉ cố định hình học và tán xạ. Lưu quãng đường từng photon đi trong từng môi trường, rồi tính lại trọng số theo định luật Beer–Lambert cho bất kỳ bộ μa nào (μa máu, độ bão hòa oxy, melanin, μa nền). | Dung lượng lưu photon; sai số so với chạy trực tiếp; chỉ đúng khi tán xạ không đổi. |
| P1b Cộng nguồn cơ sở | Mô phỏng riêng từng nguồn cơ sở (từng LED hoặc từng vị trí chiếu), rồi cộng có trọng số để ra mọi cấu hình chiếu sáng. | Tính tuyến tính (bước 01.5); số nguồn cơ sở cần thiết. |
| P2 Mô hình thay thế (emulator) | Mạng học ánh xạ θ → biên dạng từ vài nghìn lượt MC; dùng để sinh dữ liệu huấn luyện, tính Jacobian và phân tích độ nhạy. | Sai số emulator phải nhỏ hơn nhiễu camera trên một bộ MC kiểm riêng. |
| P3 Kết hợp | P1 để nhân số mẫu theo hấp thụ, P2 để làm mịn theo hình học. | — |

Ngân sách mục tiêu: khoảng 2.000–3.000 lượt MC gốc (hình học × tán xạ), mỗi lượt vài phút, tức vài ngày GPU trên RTX 3060. Nếu pilot cho thấy vượt xa, cân nhắc GPU mạnh hơn hoặc thuê GPU đám mây (vẫn dùng MCX-CL hoặc MCX với cùng kiểm chứng tương đương).

### bước 03 Sinh dữ liệu huấn luyện và bộ kiểm chứng khóa

| Bộ | Nội dung | Quy tắc |
| --- | --- | --- |
| Train / validation | Lấy mẫu θ trong khoảng sinh lý (bảng tham số.5) bằng thiết kế phủ đều không gian như Latin hypercube; chia theo hình học gốc để không rò rỉ giữa các tập. | Không chứa bất kỳ lượt nào của các bộ kiểm. |
| Kiểm nội suy | MC trực tiếp, không qua P1/P2, ở các θ mới trong khoảng huấn luyện. | Khóa manifest trước khi huấn luyện. |
| Kiểm ngoại suy | θ ngoài khoảng huấn luyện cho từng tham số (ví dụ μa thấp hơn, melanin cao hơn, D ngoài khoảng). | Báo riêng; dùng để kiểm ước lượng độ bất định có báo "không chắc chắn" ở đây không. |
| Kiểm vùng khó | Lưới dày ở vùng sâu + hấp thụ yếu. | Dùng cho H1–H2. |

Các bộ kiểm dùng MC trực tiếp, cùng tiêu chí hội tụ và chọn dữ liệu theo chất lượng MC trước khi mở dự đoán, như V1.

### bước 04 Mô hình, đối chứng và ablation

**Mô hình đề xuất:** CNN nhận C × 128 (C ảnh 850 nm ở các cấu hình đo chọn ở bước 02.6; C = 1 nếu không cấu hình nào đáng thêm), ước lượng đồng thời z và μa (có thể thêm μs′), đầu ra có phân bố (trung bình và phương sai, hàm mất mát log-likelihood Gauss), kết hợp ensemble từ ít nhất 5 mô hình để ước lượng độ bất định.

**Đối chứng** (cùng bộ kiểm khóa):

| Mã | Phương pháp |
| --- | --- |
| B1, B2 | Đặc trưng tương phản / FWHM: nghịch đảo một mức và hồi quy đa mức (từ kế hoạch luận văn). |
| B3 | Khớp mô hình thuận: bình phương tối thiểu phi tuyến của emulator với biên dạng đo, ước lượng đồng thời z, μa, μs′. Đây là đối chứng dựa trên vật lý, mạnh hơn nhiều so với K. |
| B4 | Nghịch đảo dựa trên lý thuyết khuếch tán hoặc xấp xỉ nhiễu loạn, nếu tài liệu ở bảng tham số cho một dạng dùng được. |
| B5 | 1–2 phương pháp đã công bố được tái hiện (bảng tham số.2). |
| F/K cũ | Sáu checkpoint luận văn, để nối với kết quả trước. |

**Ablation:**
- Cấu hình đo: chỉ ảnh chiếu đồng đều / thêm chiếu lệch / thêm phân cực / tổ hợp.
- Có / không dùng thông tin tiên nghiệm về μa.
- Đa dạng hóa từng tham số: bật / tắt cho μa, μs′, D, melanin.
- Đầu vào: biên dạng 1D so với ảnh 2D; kiến trúc 1D CNN so với MLP và CNN 2D.
- Cỡ tập huấn luyện: đường cong học theo số mẫu.
- Có / không ước lượng đồng thời μa.

**Thống kê:** ít nhất 10 seed cho mỗi cấu hình; khoảng tin cậy bootstrap theo **điều kiện vật lý**, không theo số ảnh hay số lượt lặp. Hiệu chuẩn độ bất định: độ phủ của khoảng dự đoán ở các mức 50/80/90/95% và sai số hiệu chuẩn.

### bước 05 Phân tích độ nhạy toàn cục và tổng hợp

| Việc | Nội dung |
| --- | --- |
| bước 05.1 Sobol | Dùng emulator tính chỉ số Sobol: phần phương sai sai số độ sâu do từng tham số và tương tác. Trả lời tham số nào gây sai số nhiều nhất. |
| bước 05.2 Bản đồ hiệu năng | Sai số và độ bất định theo (z, μa, melanin) cho mô hình đề xuất và các đối chứng. |
| bước 05.3 Đối chiếu CRLB | Sai số thực của từng phương pháp so với CRLB: phương pháp nào gần giới hạn lý thuyết. |
| bước 05.4 Gói khoa học mở | Mã mô hình thuận, emulator, bộ dữ liệu (hoặc mã sinh lại), checkpoint, manifest và hash. |

## 4 Nhu cầu tính toán

| Hạng mục | Ước tính trên RTX 3060 | Ghi chú |
| --- | --- | --- |
| Kiểm chứng mô hình thuận bước 01 | 1–3 ngày GPU | Voxel mịn có thể chậm hơn nhiều lần |
| Jacobian và CRLB bước 02 | 2–5 ngày GPU | Phụ thuộc số điểm θ và số cấu hình đo |
| Lượt MC gốc bước 02–bước 03 | 3–10 ngày GPU | Nếu phương án P1 khả thi |
| Bộ kiểm khóa bước 03 | 2–4 ngày GPU | MC trực tiếp |
| Huấn luyện bước 04 | CPU hoặc GPU; vài ngày | Ensemble × 10 seed × các ablation |

Các con số này là ước lượng cần kiểm lại bằng pilot ở bước 01/bước 02. Nếu tổng vượt khoảng một tháng GPU, cần GPU mạnh hơn.

## 5 Rủi ro chính

| Rủi ro | Cách xử lý |
| --- | --- |
| Lớp biểu bì mỏng làm mô phỏng quá đắt | So voxel mịn với MMC ở pilot; chọn phương án rẻ hơn đạt cùng tiêu chí hội tụ. |
| Phương án P1 không khả thi về lưu trữ | Dùng emulator (P2) với nhiều lượt MC hơn; giảm chiều không gian tham số theo kết quả Sobol sơ bộ. |
| Emulator đưa sai số hệ thống vào dữ liệu | Mọi bộ kiểm dùng MC trực tiếp; báo sai số emulator riêng. |
| H1 không được hỗ trợ | Đổi trọng tâm như cổng bước 02; vẫn là kết quả có giá trị. |
| Không cấu hình 850 nm nào thêm đủ thông tin | Trọng tâm chuyển sang giới hạn lý thuyết và ước lượng độ bất định, như cổng bước 02. |
| `pmcxcl` không mô phỏng được phân cực | Bỏ phương án phân cực hoặc dùng một công cụ Monte Carlo phân cực đã được kiểm chứng, sau khi kiểm tương đương với `pmcxcl` ở trường hợp không phân cực. |
| Tham số da trong tài liệu dao động lớn | Dùng khoảng rộng và báo độ nhạy, không chọn một bộ số rồi coi là "da người". |

## 6 Liên hệ với phần thực nghiệm

Kết quả mô phỏng này quyết định thiết kế phần thực nghiệm sau: cấu hình chiếu sáng và phân cực cần lắp trên hệ 850 nm (bước 02.6), khoảng tham số phantom cần chế tạo, và độ chính xác kỳ vọng trên người. Phantom đo tính chất quang bằng quả cầu tích phân và nghiên cứu trên người (phê duyệt đạo đức, siêu âm tham chiếu, đa dạng màu da) được lập kế hoạch riêng sau khi bước 02 hoàn tất.
