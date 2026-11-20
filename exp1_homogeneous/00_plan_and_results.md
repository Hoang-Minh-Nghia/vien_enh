# Kế hoạch mô phỏng và so sánh CNN F K

Giai đoạn tiếp theo được mô tả trong [Kế hoạch kiểm chứng độc lập F/K](Ke_hoach_tiep_theo_kiem_chung_doc_lap_FK.md): giữ nguyên sáu checkpoint, thử bằng MC trực tiếp và mức hấp thụ chưa huấn luyện, sau đó kiểm chứng phantom. Đây là kế hoạch mới, chưa phải các kết quả đã chạy dưới đây.

## Tiến độ thực hiện

| Giai đoạn | Trạng thái thực tế | Kết quả hoặc vị trí kiểm tra |
| --- | --- | --- |
| 1 Mô phỏng thử | Đã hoàn thành | N0 120 triệu photon, kênh 2 mm 60 triệu photon; năm phép kiểm đều đạt. Báo cáo: `exp1_homogeneous/01_simulator_checks/results/report.json`; hình: `exp1_homogeneous/01_simulator_checks/results/stage1_overview.png`. |
| 2 Dữ liệu nhiều độ sâu | Đã hoàn thành | 12 độ sâu ở μa = 0,508 mm⁻¹; ba cấu hình kiểm nội suy 0,625 / 2,25 / 4,75 mm đều đạt. Xem `exp1_homogeneous/02_forward_model/results/report.json` và `depth_profiles.png`. |
| 3 Chạy thử CNN | Đã hoàn thành | Bộ thử 1.200 cảnh, chia 960/120/120. Seed 42, chọn epoch 76 bằng validation. MAE test 0,06377 mm; đối chứng đoán trung bình train: 1,12662 mm. Xem `exp1_homogeneous/04_training/results/pilot_K/trial_report.json` và `trial_overview.png`. |
| 4 So sánh F/K | Đã hoàn thành | Đủ 48 cấu hình kênh chính, năm kiểm mở rộng đều đạt; hai bộ 6.000 mẫu × 128 điểm, sáu checkpoint cho F/K với seed 42/43/44. Xem `exp1_homogeneous/04_training/results/FK/comparison.json`, `comparison.csv` và `comparison.png`. |

### Kết quả so sánh chính thức trong mô phỏng

MAE dưới đây là trung bình của ba seed, cùng chấm trên 600 đầu vào test F. Nhóm hấp thụ yếu gồm 300 cảnh; nhóm chuẩn và cao có 150 cảnh mỗi nhóm.

| Nhóm thử | MAE F mm | MAE K mm | Nhận xét |
| --- | --- | --- | --- |
| Hấp thụ yếu 0,15 và 0,30 mm⁻¹ | 0,262895 | 0,621701 | F giảm 0,358807 mm, tương đương 57,71% MAE so với K. |
| Hấp thụ chuẩn 0,508 mm⁻¹ | 0,057578 | 0,026214 | K chính xác hơn ở điều kiện chuẩn; F tăng MAE khoảng 0,031365 mm. |
| Hấp thụ cao 1,0 mm⁻¹ | 0,086157 | 0,249478 | F có MAE thấp hơn trong nhóm này. |
| Toàn bộ test F | 0,167381 | 0,379774 | Tổng hợp cả bốn mức hấp thụ; không thay thế chỉ số chính của nhóm yếu. |

| Seed huấn luyện | MAE F nhóm yếu mm | MAE K nhóm yếu mm |
| --- | --- | --- |
| 42 | 0,169371 | 0,597603 |
| 43 | 0,298249 | 0,653451 |
| 44 | 0,321064 | 0,614050 |

F giảm MAE nhóm yếu ở cả ba seed. Độ lệch chuẩn MAE giữa seed là 0,081793 mm cho F và 0,028699 mm cho K; đây là biến thiên huấn luyện trên cùng tập thử, không phải độ bất định thực nghiệm.

Mục tiêu MAE ≤ 0,5 mm chưa đạt ở mọi nhóm: tại μa = 0,15 mm⁻¹, F còn có MAE trung bình khoảng 0,59 mm ở khoảng sâu 3–3,5 mm và 0,70 mm ở 3,5–4 mm. Xem `exp1_homogeneous/04_training/results/FK/mae_depth_absorption.png` và các số đếm cảnh trong `comparison.json`.

Kiểm tra cuối đã đạt: chín kiểm tra chương trình; đối chiếu 48 nguồn MC, chia tập và cân bằng hấp thụ, sáu mô hình đúng bộ dữ liệu, đầu vào F/K trùng nhau ở hấp thụ chuẩn; tính lại MAE nhóm yếu từ CSV dự đoán khớp báo cáo trong sai số 10⁻⁶ mm. Hồ sơ đối chiếu: `exp1_homogeneous/maintenance/audit.json`.

Kết quả bước 1 sau khi thống nhất phép chia N0 tại từng điểm trước khi lấy trung bình: sai khác tương phản / độ rộng lần lượt là 0,41% / 1,92% khi đổi seed; 0,15% / 0,62% khi tăng gấp đôi photon; 0,84% / 0,05% khi làm mịn lưới; 0,39% / 0,40% khi mở rộng miền ngang; 1,15% / 0,93% khi tăng thời gian. Các giá trị đều dưới ngưỡng 10% ở cấu hình kênh 2 mm. Đây là kiểm tra số học, chưa phải sai số dự đoán độ sâu của CNN.

Lượt lưới 0,05 mm với batch 6 triệu photon từng trả NaN trên môi trường hiện có. Đã chạy lại thành công với batch 1 triệu photon, giữ nguyên tổng photon. Dữ liệu lỗi được tách riêng; không dùng để huấn luyện. Xem `README.md` và `exp1_homogeneous/maintenance/diagnostics`.

Sai số CNN bước 3 chỉ mô tả bộ thử phát triển ở hấp thụ chuẩn, tạo bằng nội suy bảng tra MC chung, quan sát lý tưởng. Không dùng con số này làm độ chính xác thực nghiệm hoặc làm kết quả so sánh F/K. Bộ chính thức dùng seed tạo cảnh riêng, tách với bộ thử phát triển.

MAE dưới kích thước voxel 0,1 mm trên nhãn nội suy không chứng minh hệ thật đạt độ phân giải dưới voxel. Kiểm nội suy bằng MC độc lập chỉ đánh giá chất lượng phép nội suy ở các cấu hình đã kiểm; test CNN vẫn được sinh từ bảng tra chung và cần được mô tả đúng phạm vi này.

Mục tiêu là thực hiện lần lượt bốn giai đoạn: mô phỏng thử, tạo dữ liệu nhiều độ sâu, chạy thử CNN và so sánh F/K. Toàn bộ công việc trong kế hoạch này có thể làm trước khi có số đo camera hoặc ảnh phantom. Dữ liệu đầu vào được tạo bằng mô phỏng; kết luận chỉ áp dụng cho các điều kiện mô phỏng đã khảo sát.

## Phạm vi và quy tắc chung

| Nội dung | Quy định |
| --- | --- |
| Câu hỏi nghiên cứu | Huấn luyện với hấp thụ kênh thay đổi có giảm sai lệch dự đoán độ sâu khi hấp thụ yếu hơn mức chuẩn không? |
| Bộ K | Hấp thụ kênh cố định ở 0,508 mm⁻¹. |
| Bộ F | Hấp thụ kênh gồm 0,15; 0,30; 0,508; 1,0 mm⁻¹, chia đều số mẫu. |
| Hệ ảnh | Phản xạ 850 nm; nguồn và mặt thu ở cùng phía. |
| Hình học | Một kênh thẳng đường kính 3 mm trong nền đồng nhất; độ sâu mép trên 0,5–5 mm. |
| Nền | μs′ = 1,0 mm⁻¹; μa = 0,010 mm⁻¹; g = 0,511; khi công cụ yêu cầu μs, dùng μs = μs′/(1 − g) ≈ 2,045 mm⁻¹. |
| Gần đúng quang học | n nền = n kênh = 1,35; tán xạ kênh bằng 0. Đây là giả định mô hình, chưa phải số đo vật liệu. |
| Miền tính ban đầu | Nền dày 20 mm; voxel 0,1 mm; nguồn phẳng; biên ngang tuần hoàn và đáy hấp thụ theo thiết kế. Kiểm khả năng biểu diễn của công cụ và ảnh hưởng biên trước khi khóa cấu hình. |
| Nhãn | Độ sâu từ mặt quan sát đến mép trên kênh, đơn vị mm; tâm kênh nằm ở độ sâu mép trên cộng 1,5 mm. |
| Đại lượng đầu ra MC | Tín hiệu phản xạ thoát qua mặt quan sát, được chuẩn hóa theo số photon và quy ước công cụ; không mặc định fluence trong thể tích chính là ảnh phản xạ camera. |
| Camera khi chưa đo thực tế | Bắt đầu với quan sát lý tưởng. Nếu thêm nhiễu hoặc độ nhòe, lưu mô hình và tham số giả định, áp dụng cùng quy tắc cho F/K. Chưa gán các tham số này là đặc tính Camera Module 3 NoIR. |
| Đầu vào CNN | Biên dạng ngang 128 điểm từ vùng rộng 32 mm, lấy trung bình dọc đoạn kênh dài 8 mm; kích thước này quy đổi trực tiếp theo tọa độ mô phỏng, chưa cần mm/pixel đo thực tế. |
| Tiền xử lý | Chia phản xạ có kênh cho N0 tương ứng. Với dữ liệu lý tưởng không có offset cảm biến, ảnh tối bằng 0. Nếu mô phỏng offset, trừ tối ở cả tử và mẫu trước khi chia. Không chuẩn hóa min–max riêng từng mẫu. |
| Nguyên tắc so sánh | F/K dùng cùng cảnh, hình học, mô hình quan sát, nguồn ngẫu nhiên, kiến trúc và ngân sách huấn luyện; khác chính sách hấp thụ kênh. |
| Giới hạn | Không xác nhận độ chính xác trên phantom, người hoặc tốc độ Raspberry Pi trong kế hoạch này. |

## Bảng công việc chi tiết

Bảng dưới đây mô tả công việc và tiêu chí cần đạt. Trạng thái thực tế được cập nhật ở bảng tiến độ đầu file; không suy ra một công việc đã hoàn thành chỉ từ việc có tên trong kế hoạch.

| Mã | Giai đoạn và công việc | Đầu vào hoặc cách thực hiện | Đầu ra phải lưu | Điều kiện hoàn thành và chuyển bước |
| --- | --- | --- | --- | --- |
| 1.1 | Mô phỏng thử — kiểm tra môi trường | Kiểm GPU, OpenCL/CUDA và lựa chọn MCX hoặc MCX-CL phù hợp; chạy ví dụ của công cụ. | Phiên bản phần mềm, thông tin phần cứng, lệnh chạy và log ví dụ. | Mô phỏng chạy được; kết quả đọc được bằng chương trình xử lý. |
| 1.2 | Mô phỏng thử — lập cấu hình chung | Lưu thông số trong bảng phạm vi; ghi rõ đơn vị, trục tọa độ, mặt quan sát và vị trí kênh. Chốt miền ngang đủ chứa ROI và khoảng đệm sau kiểm tra biên. | File cấu hình, sơ đồ tọa độ và bảng vật liệu. | Kiểm được đường kính, độ sâu mép trên, tâm kênh và quy đổi μs′ sang μs; không nhầm đơn vị voxel với mm. |
| 1.3 | Mô phỏng thử — thu nền N0 | Chạy khối không kênh; ngân sách khởi đầu 1,2 × 10⁸ photon; xác định cách ghi nhận photon phản xạ ở mặt trên. | Cấu hình N0, seed, log, bản đồ phản xạ và thông tin chuẩn hóa. | Giá trị hữu hạn, có tín hiệu trong ROI; phân bố phù hợp nguồn và điều kiện biên đã đặt. |
| 1.4 | Mô phỏng thử — chạy kênh 2 mm | Thêm kênh D = 3 mm, độ sâu mép trên 2 mm, μa = 0,508 mm⁻¹; ngân sách khởi đầu 6 × 10⁷ photon. | Cấu hình, bản đồ phản xạ, hình kiểm mặt cắt hình học. | Kênh đúng vị trí; cách thu và chuẩn hóa thống nhất với N0. |
| 1.5 | Mô phỏng thử — trích biên dạng | Tính R_kênh/R_N0; kiểm các điểm N0 quá nhỏ; lấy trung bình dọc 8 mm và lấy mẫu ngang 32 mm về 128 điểm. | Biên dạng gốc, biên dạng chuẩn hóa, hình đối chiếu và quy tắc xử lý mẫu không hợp lệ. | Có 128 giá trị hữu hạn; vị trí kênh phù hợp; giữ được tương phản, không dùng min–max từng ảnh. |
| 1.6 | Mô phỏng thử — kiểm hội tụ và biên | Chạy seed MC độc lập; tăng photon khi cần; so lưới ban đầu với lưới mịn hơn; tăng miền ngang hoặc đổi cấu hình biên để kiểm ảnh hưởng. | Bảng so tương phản, độ rộng và sai khác biên dạng; thời gian chạy; cấu hình được chọn. | Ở cấu hình đủ tín hiệu, sai khác tương phản và độ rộng đạt ngưỡng dự kiến ≤ 10%; lưu định nghĩa phép đo. Với tín hiệu yếu, bổ sung sai số tuyệt đối và không ép tính độ rộng khi không xác định được. |
| 1.7 | Mô phỏng thử — khóa bản chạy tối thiểu | Tổng hợp các kiểm tra trên; lưu cấu hình và script tái tạo N0 cùng kênh 2 mm. | Một bộ kết quả mẫu có thể chạy lại và báo cáo kiểm tra ngắn. | Có thể tái tạo kết quả trong giới hạn dao động MC; chỉ chuyển bước khi lỗi hình học, mặt thu và chuẩn hóa đã được xử lý. |
| 2.1 | Tạo dữ liệu nhiều độ sâu — lập lịch chạy | Bắt đầu ở hấp thụ chuẩn 0,508 mm⁻¹; dùng 12 độ sâu: 0,5; 0,75; 1; 1,25; 1,5; 2; 2,5; 3; 3,5; 4; 4,5; 5 mm. | Danh sách cấu hình, mã chạy và seed riêng. | Đủ 12 độ sâu, đơn vị và định nghĩa nhãn thống nhất. |
| 2.2 | Tạo dữ liệu nhiều độ sâu — xây bảng tra K thử | Chạy các cấu hình; dùng N0 tương thích; kiểm thêm độ sâu nông và sâu thay vì suy rộng kiểm tra 2 mm cho mọi trường hợp. | Kết quả MC, bảng tra biên dạng theo độ sâu và báo cáo chất lượng. | Cấu hình được dùng đều đạt kiểm tra đã định; đánh dấu vùng tín hiệu quá yếu và tăng photon nếu cần. |
| 2.3 | Tạo dữ liệu nhiều độ sâu — kiểm nội suy | Chạy độc lập tại một số độ sâu giữa các mốc; so với biên dạng nội suy. Nếu chưa đạt, thêm mốc hoặc sửa phương pháp nội suy. | Cấu hình kiểm riêng, sai số nội suy và phiên bản bảng tra. | Dùng tiêu chí tương phản/độ rộng đã khóa khi đủ tín hiệu; không ngoại suy ngoài 0,5–5 mm. Không đưa cấu hình kiểm riêng vào train khi đang dùng nó để xác minh nội suy. |
| 2.4 | Tạo dữ liệu nhiều độ sâu — định nghĩa cảnh và chia tập | Mỗi scene_id gắn với độ sâu, vị trí và các yếu tố mô phỏng đã chọn. Lập danh sách chia tập trước khi sinh biến thể nhiễu. | Manifest cảnh, train/validation/test và seed tạo cảnh. | Mọi biến thể của cùng cảnh nằm trong cùng một tập. Ghi rõ nếu các cảnh cùng dùng một bảng tra MC, vì chúng không phải các mô phỏng vật lý độc lập. |
| 2.5 | Tạo dữ liệu nhiều độ sâu — sinh bộ thử nhỏ | Sinh số lượng vừa đủ để kiểm quy trình, không bắt buộc chạy ngay 6.000 mẫu. Dùng quan sát lý tưởng trước; dữ liệu có nhiễu giả định được quản lý thành phiên bản riêng. | X dạng N × 128 float32; y độ sâu mm; manifest và cấu hình quan sát. | Nhãn đúng, không NaN/Inf, phân bố độ sâu được kiểm; quy trình tạo dữ liệu có thể lặp lại. |
| 2.6 | Tạo dữ liệu nhiều độ sâu — kiểm dữ liệu | Vẽ biên dạng ở các độ sâu, thống kê nhãn, kiểm mã cảnh trùng và đối chiếu ngẫu nhiên đầu vào với nhãn. | Báo cáo kiểm dữ liệu và các hình kiểm tra. | Không có cảnh bị chia qua nhiều tập; các điểm bất thường được xử lý bằng quy tắc chất lượng, không dựa vào lỗi dự đoán CNN. |
| 3.1 | Chạy thử CNN — lập mô hình | CNN 1D với ba khối tích chập 16/32/64 kênh, kernel 5, ReLU; gộp trung bình toàn cục; lớp ẩn 32; đầu ra một độ sâu. Ghi rõ padding, stride và các lựa chọn triển khai. | Mã mô hình, cấu hình và phép kiểm kích thước đầu vào/đầu ra. | Nhận batch × 1 × 128 và trả một giá trị mỗi mẫu; không đưa μa hoặc nhãn vào đầu vào. |
| 3.2 | Chạy thử CNN — huấn luyện K thử | Adam, learning rate 0,001, batch 64, MAE; tối đa 100 epoch; dừng sớm sau 10 epoch validation không cải thiện. Có thể dùng seed 42 cho lần kiểm quy trình đầu. | Log train/validation, checkpoint tốt nhất theo validation và cấu hình chạy. | Loss hữu hạn, mô hình cập nhật được; checkpoint được chọn hoàn toàn bằng validation. |
| 3.3 | Chạy thử CNN — đối chiếu dự đoán | Chạy test giữ riêng; so với cách luôn dự đoán trung bình nhãn train; kiểm biểu đồ dự đoán–độ sâu thật và sai số theo độ sâu. | MAE, RMSE, bias; bảng dự đoán; đường học và biểu đồ sai số. | Quy trình đánh giá chạy đúng. Nếu CNN không vượt cách đoán đơn giản, cần kiểm dữ liệu/kiến trúc/huấn luyện trên train và validation trước khi mở rộng. Không coi một ngưỡng MAE tự chọn là bằng chứng khớp ảnh thật. |
| 3.4 | Chạy thử CNN — khóa quy trình | Chốt tiền xử lý, kiến trúc, cách chọn checkpoint và quy tắc đánh giá cho phép so sánh chính thức. Tập test đã xem để phát triển được xem là test thử. | Phiên bản mã và cấu hình đã khóa; ghi chú các thay đổi đã thực hiện. | Tạo tập test chính thức chưa dùng để điều chỉnh ở giai đoạn 4; bảo đảm kết quả thử không bị trình bày như kiểm định cuối. |
| 4.1 | So sánh F/K — mở rộng bảng tra | Chạy 12 độ sâu × 4 mức μa = 48 cấu hình kênh và N0 phù hợp; tái sử dụng kết quả chuẩn đã đạt nếu cấu hình không đổi. | Bảng tra F/K có phiên bản, log MC và danh sách cấu hình. | Kiểm hội tụ, biên và nội suy ở các cấu hình đại diện, gồm hấp thụ yếu và độ sâu lớn; không chỉ dựa vào phép kiểm tại μa chuẩn. |
| 4.2 | So sánh F/K — tạo manifest ghép cặp | Tạo 6.000 mã cảnh có cùng độ sâu, vị trí và tham số quan sát cho hai nhánh; F chia đều bốn hấp thụ, K luôn 0,508 mm⁻¹. | Manifest chung, hấp thụ từng nhánh và seed. | Theo từng scene_id, khác biệt chủ đích là hấp thụ kênh; không vô tình thay nền, D hoặc phân bố độ sâu giữa F/K. |
| 4.3 | So sánh F/K — sinh và chia bộ chính thức | Mỗi bộ 6.000 mẫu: 4.800 train, 600 validation, 600 test. Trong F, mỗi μa có 1.200/150/150 mẫu. Dùng chia tập theo cảnh trước khi sinh nhiễu. | Hai bộ dữ liệu, nhãn, split, cấu hình quan sát và báo cáo kiểm dữ liệu. | Đúng số lượng, chia đều mức hấp thụ và phân bố độ sâu tương đương. Nếu quan sát có nhiễu phụ thuộc tín hiệu, cùng nguồn ngẫu nhiên không đồng nghĩa cộng cùng một mảng nhiễu có biên độ cố định. |
| 4.4 | So sánh F/K — huấn luyện sáu lần | Huấn luyện F và K cho mỗi seed 42, 43, 44; cùng khởi tạo trong mỗi cặp seed, cùng siêu tham số và ngân sách; mỗi nhánh chọn checkpoint bằng validation tương ứng. | Sáu checkpoint, sáu log và cấu hình tái lập. | Không dùng test để chọn epoch, checkpoint hoặc sửa siêu tham số. |
| 4.5 | So sánh F/K — chấm cùng tập test F | Với từng seed, cả F và K dự đoán đúng 600 đầu vào test F: 300 hấp thụ yếu, 150 chuẩn, 150 cao. Test K dùng kiểm bổ sung điều kiện chuẩn. | Bảng scene_id, μa, độ sâu thật, dự đoán F/K và sai số; lưu seed mô hình. | Hai nhánh dùng chính cùng đầu vào test, không chỉ hai tập có số lượng tương đương. |
| 4.6 | So sánh F/K — tổng hợp và kết luận | Chỉ số chính: MAE nhóm μa = 0,15 và 0,30. Bổ sung RMSE, bias = dự đoán − nhãn, kết quả theo độ sâu và μa; báo cả ba seed. | Bảng chỉ số, biểu đồ và chênh lệch ΔMAE = MAE_K − MAE_F; ΔMAE dương biểu thị F tốt hơn về MAE. | Báo cả trường hợp F không tốt hơn hoặc giảm độ chính xác ở mức chuẩn. Phân biệt biến thiên giữa seed huấn luyện với tính độc lập của dữ liệu mô phỏng. |
| 4.7 | So sánh F/K — hoàn tất hồ sơ | Lưu cấu hình, phiên bản công cụ, seed, bảng tra, dữ liệu, split, checkpoint và báo cáo. Nếu khảo sát nhiễu/độ nhòe giả định, báo riêng từng điều kiện. | Bộ kết quả tái lập và danh sách giới hạn nghiên cứu. | Người khác có thể chạy lại phép so sánh; kết luận ghi rõ chỉ có hiệu lực trong miền mô phỏng, chưa được xác nhận bằng camera/phantom. |

## Các mốc bàn giao

| Mốc | Hoàn thành sau | Hồ sơ tối thiểu | Quyết định tiếp theo |
| --- | --- | --- | --- |
| M1 | Giai đoạn 1 | N0, kênh 2 mm, biên dạng chuẩn hóa và báo cáo kiểm hội tụ ban đầu. | Nếu đúng và ổn định thì chạy nhiều độ sâu; nếu chưa thì sửa mô phỏng. |
| M2 | Giai đoạn 2 | Bảng tra nhiều độ sâu đã kiểm nội suy, bộ dữ liệu thử và manifest chia tập. | Chuyển sang thử CNN khi dữ liệu đạt kiểm tra. |
| M3 | Giai đoạn 3 | CNN thử, checkpoint, log và bảng đánh giá có đối chứng đơn giản. | Khóa quy trình rồi mở rộng sang F/K. |
| M4 | Giai đoạn 4 | Hai bộ 6.000 mẫu, sáu checkpoint và kết quả ghép cặp trên test chung. | Kết luận trong mô phỏng; sau đó mới hiệu chuẩn hệ thật và kiểm phantom theo kế hoạch riêng. |

## Kiểm soát các lỗi có thể làm sai kết luận

| Rủi ro | Cách kiểm soát |
| --- | --- |
| Nhầm fluence với phản xạ bề mặt | Xác định và lưu đúng đại lượng thu trước khi tạo bảng tra. |
| Nhầm độ sâu tâm với độ sâu mép trên | Kiểm hình học bằng mặt cắt và công thức tâm = mép trên + D/2. |
| Nhầm μs với μs′ | Quy đổi theo g và lưu cả hai giá trị trong cấu hình. |
| Nhiễu Monte Carlo trở thành đặc trưng mà CNN học | Kiểm hội tụ; dùng các lần MC độc lập để kiểm tra độ bền khi cần; ghi rõ việc dùng chung bảng tra. |
| Các bản nhiễu cùng cảnh xuất hiện ở nhiều tập | Chia scene_id trước khi sinh biến thể; kiểm giao của các tập bằng mã cảnh. |
| Nội suy làm dữ liệu quá thuận lợi | Kiểm bằng mô phỏng độc lập ở độ sâu trung gian và mô tả rõ test nào sinh từ nội suy. |
| F và K khác thêm yếu tố ngoài hấp thụ | Dùng manifest ghép cặp và kiểm tự động các trường phải giống nhau. |
| Loại ảnh vì dự đoán sai hoặc điều chỉnh theo test | Khóa quy tắc chất lượng trước; chọn mô hình bằng validation; giữ test chính thức riêng. |
| Báo mục tiêu thành kết quả | MAE ≤ 0,5 mm là mục tiêu khảo sát; chỉ báo đạt khi có số đo trên tập thử và nêu rõ miền điều kiện. |
| Suy rộng kết quả giả lập ra hệ thật | Tách kết luận mô phỏng khỏi kết luận thực nghiệm; hiệu chuẩn camera và kiểm phantom là giai đoạn sau. |

## Thứ tự thực hiện

Thực hiện 1.1–1.7 trước, sau đó 2.1–2.6, tiếp đến 3.1–3.4 và cuối cùng 4.1–4.7. Không cần chờ số đo thực tế để hoàn thành bốn giai đoạn. Chưa cần tạo ảnh camera 768 × 576 nếu đầu vào CNN có thể được tạo trực tiếp từ phản xạ bề mặt theo đơn vị vật lý; mọi bước chuyển từ mô phỏng sang ảnh camera phải được mô tả khi bổ sung mô hình quan sát.

Các thông số thiết kế kế thừa từ `Ke_hoach_phan_xa_FK_nen_quang_hoc_co_dinh.docx` trong cùng thư mục. Những lựa chọn triển khai bổ sung như miền ngang, cách ghi nhận phản xạ và mô hình nhiễu giả định phải được xác nhận qua các phép kiểm ở trên trước khi khóa bộ dữ liệu chính thức.
