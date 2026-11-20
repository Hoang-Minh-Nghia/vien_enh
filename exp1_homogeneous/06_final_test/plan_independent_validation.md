# Kế hoạch kiểm chứng độc lập khả năng dự đoán độ sâu của F và K

Kế hoạch tiếp theo kiểm tra lợi ích của F trên các lượt Monte Carlo mới, các mức hấp thụ chưa dùng huấn luyện và cuối cùng trên phantom. Giữ nguyên sáu checkpoint hiện có để đánh giá khả năng khái quát của mô hình đã huấn luyện. **Trạng thái:** V0 đã hoàn thành, phép kiểm được khóa trong `configs/validation_independent.json` (SHA-256 ở `configs/validation_independent.sha256`, tạo bằng `code/v0_lock_manifest.py`, kiểm bằng `code/test_validation_manifest.py`). V1–V8 **chưa được thực hiện**; các phần còn lại của file này là kế hoạch, không phải kết quả mới.

## 1 Căn cứ và câu hỏi cần trả lời

| Nội dung | Hiện đã có | Khoảng trống cần kiểm chứng |
| --- | --- | --- |
| Dữ liệu K | 6.000 biên dạng 128 điểm; μa kênh cố định 0,508 mm⁻¹. | Mô hình phản ứng thế nào khi hấp thụ thật khác mức chuẩn? |
| Dữ liệu F | 6.000 biên dạng 128 điểm; μa kênh gồm 0,15 / 0,30 / 0,508 / 1,0 mm⁻¹. | Lợi ích có giữ được trên MC mới và mức μa chưa dùng huấn luyện không? |
| Kết quả nhóm yếu hiện tại | MAE trung bình ba seed: F = 0,262895 mm; K = 0,621701 mm; F giảm 57,71%. | Test hiện được nội suy từ cùng bảng tra dùng tạo dữ liệu học, chưa phải phép thử CNN bằng MC độc lập. |
| Đánh đổi ở mức chuẩn | MAE F = 0,057578 mm; K = 0,026214 mm. | Cần tiếp tục định lượng đánh đổi trên dữ liệu kiểm chứng mới, không chỉ báo nhóm F có lợi. |
| Miền chưa đạt | F còn vượt MAE 0,5 mm ở μa = 0,15 mm⁻¹, các khoảng sâu 3–3,5 và 3,5–4 mm trên test cũ. | Kiểm lại vùng này bằng MC trực tiếp và ghi rõ nếu giới hạn vẫn tồn tại. |
| Quan sát | Phản xạ bề mặt lý tưởng; chưa có nhiễu, độ nhòe và đáp ứng camera đo thực tế. | Chưa xác nhận được hiệu quả trên hệ chụp thật hoặc phantom. |

Câu hỏi chính: **F có giảm sai số và giảm sự phụ thuộc của dự đoán độ sâu vào hấp thụ trên dữ liệu mới, trong khi vẫn phân biệt được các độ sâu khác nhau không?** Không đặt trước yêu cầu F phải thắng ở mọi điều kiện.

## 2 Các thành phần phải khóa trước khi chạy

| Thành phần | Quy định |
| --- | --- |
| Sáu mô hình | Giữ `cnn_F_seed42.pt`, `cnn_F_seed43.pt`, `cnn_F_seed44.pt`, `cnn_K_seed42.pt`, `cnn_K_seed43.pt`, `cnn_K_seed44.pt` trong `exp1_homogeneous/04_training/results/FK/models`. Lập danh sách đường dẫn và SHA-256 trước khi kiểm chứng. |
| Kiến trúc và trọng số | Không huấn luyện lại, chọn lại checkpoint, đổi kiến trúc hoặc điều chỉnh siêu tham số bằng bộ kiểm chứng mới. |
| Tiền xử lý | Giữ phiên bản `pixelwise_N0_then_y_mean_then_x_bin/v1`: chia N0 tại từng điểm, lấy trung bình dọc kênh 8 mm, trung bình theo 128 ô ngang trong vùng 32 mm. Không min–max từng mẫu. |
| Hình học và quang học | Giữ cấu hình `configs/simulation.json`: một kênh D = 3 mm; độ sâu mép trên; nền cố định; 850 nm; g = 0,511; cùng nguồn và biên đã kiểm. |
| Quan sát của kiểm chứng MC | Vẫn là phản xạ bề mặt lý tưởng; không tự thêm tham số camera chưa đo rồi gọi là hệ thật. |
| Chọn vị trí kênh | Bộ kiểm chính đặt kênh ở giữa miền và ROI theo quy tắc cố định. Kiểm dịch ngang ±2 mm có thể làm bổ sung, phải báo riêng và không tăng số điều kiện MC độc lập bằng các bản dịch của cùng trường. |
| Danh sách cấu hình | Lưu toàn bộ độ sâu, μa, seed MC, số photon, tiêu chí chất lượng và công thức tính chỉ số trước khi xem dự đoán CNN. |
| Test đã dùng | Test cũ và các phép kiểm nội suy cũ được ghi là dữ liệu đã xem. Không đổi tên chúng thành bộ kiểm chứng mới. |
| Nếu phát hiện cần sửa mô hình | Bộ mới trở thành dữ liệu phát triển kể từ lúc dùng để điều chỉnh. Phiên bản mô hình sau sửa cần một bộ kiểm chứng khác chưa dùng điều chỉnh. |

## 3 Thiết kế bộ Monte Carlo kiểm chứng mới

### 3.1 Ma trận điều kiện chính

Các giá trị sau là lựa chọn thiết kế cho giai đoạn mới, không phải kết quả đã đo.

| Trục khảo sát | Giá trị dự kiến | Mục đích |
| --- | --- | --- |
| Độ sâu mép trên mm | 0,6; 0,9; 1,4; 1,8; 2,3; 2,8; 3,3; 3,8; 4,3; 4,8 | Mười độ sâu không trùng 12 nút bảng tra huấn luyện và không trùng ba độ sâu kiểm nội suy chuẩn cũ. Có điểm trong vùng F từng sai nhiều. |
| Hấp thụ yếu đã có trong F | 0,15 và 0,30 mm⁻¹ | Kiểm chuyển từ test nội suy sang MC mới ở các điều kiện F đã học. |
| Hấp thụ chưa dùng huấn luyện | 0,22 và 0,40 mm⁻¹ | Kiểm khái quát theo μa. Hai mức nằm trong khoảng huấn luyện, nên đây là kiểm mức trung gian, chưa phải kiểm ngoại suy ngoài khoảng. |
| Hấp thụ đối chứng | 0,508 và 1,0 mm⁻¹ | Định lượng đánh đổi tại mức chuẩn và đánh giá mức cao. |
| MC lặp độc lập | Ba seed mới cho mỗi cặp độ sâu–μa | Kiểm ảnh hưởng của nhiễu ước lượng MC; các seed này khác seed huấn luyện CNN 42/43/44. |
| N0 | Ba lượt MC nền mới, mỗi lượt gắn với một chỉ số lặp | Không dùng N0 của bảng tra cũ làm nền duy nhất cho bộ kiểm chứng. F/K cùng dùng chính đầu vào đã chuẩn hóa của từng lượt. |

Tổng cộng: **10 độ sâu × 6 mức hấp thụ = 60 điều kiện vật lý mô phỏng**, mỗi điều kiện có ba lượt MC, tương ứng **180 bản đồ phản xạ có kênh**, cộng **3 bản đồ N0**. Các bản đồ MC này được trích thành biên dạng 128 điểm để đưa vào CNN; không gọi chúng là hai bộ 6.000 ảnh camera mới.

Mỗi trong sáu CNN dự đoán trên cùng 180 biên dạng, tạo 1.080 giá trị dự đoán. Đây **không phải 1.080 mẫu độc lập**: chúng bao gồm cùng điều kiện chạy lặp MC và cùng đầu vào được nhiều mô hình dự đoán.

### 3.2 Ngân sách photon và điều kiện chất lượng

| Nội dung | Quy định trước khi chạy |
| --- | --- |
| Ngân sách khởi đầu | 6 × 10⁷ photon cho mỗi lượt có kênh; 1,2 × 10⁸ photon cho mỗi lượt N0. Đây là ngân sách ban đầu, chưa bảo đảm mọi cấu hình đều hội tụ. |
| Pilot | Chạy trước z = 0,6 / 3,3 / 4,8 mm tại μa = 0,15 / 0,508 mm⁻¹, cùng N0 tương ứng. Dùng để kiểm chất lượng số trước khi mở dự đoán CNN. Nếu cấu hình không đổi và đạt, tái sử dụng các lượt pilot này trong ma trận chính. |
| Tính hợp lệ | Kiểm seed, photon thực, tổng năng lượng ban đầu, phản xạ hữu hạn, N0 dương, đúng lớp thu và đúng hình học. Không sửa NaN hoặc bỏ điểm lỗi để cho CNN chạy tiếp. |
| Hội tụ | Dùng tiêu chí đã có: sai khác tương phản và FWHM ≤ 10% khi đủ tín hiệu; với tương phản tham chiếu < 0,05, dùng ngưỡng chênh lệch tương phản tuyệt đối 0,005, đồng thời kiểm độ rộng. Chọn lượt tham chiếu theo chỉ số lặp cố định, không chọn sau khi xem CNN. |
| Khi chưa đạt | Tăng photon theo các mức 6 × 10⁷ → 1,2 × 10⁸ → 2,4 × 10⁸ cho cả ba lượt của điều kiện chưa đạt; lưu mọi phiên bản và quyết định dựa trên chất lượng MC. Nếu vẫn chưa đạt, ghi điều kiện đó là chưa đủ độ tin cậy; không tuyên bố đã kiểm chứng đầy đủ miền này. |
| Lưới và biên | Kiểm bổ sung ở cấu hình nông và cấu hình sâu, hấp thụ yếu nếu độ rộng hoặc tương phản bất thường. Nếu cần đổi cấu hình số, xác định lại cách so với cấu hình chuẩn và lưu riêng phiên bản. |
| Chọn dữ liệu dùng đánh giá | Khóa lựa chọn theo chất lượng MC trước khi xem dự đoán. Không loại cấu hình vì CNN sai nhiều và không thử nhiều seed để chỉ giữ lượt thuận lợi. |

Các lượt kiểm mới được lưu riêng khỏi `exp1_homogeneous/01_simulator_checks/results`–`exp1_homogeneous/04_training/results/FK`. Không nối chúng vào bảng tra hoặc dữ liệu train của sáu mô hình đang được kiểm chứng.

## 4 Bảng công việc theo thứ tự

| Mã | Việc thực hiện | Cách làm | Đầu ra phải lưu | Điều kiện hoàn thành |
| --- | --- | --- | --- | --- |
| V0 | Khóa phép kiểm | Lập manifest 60 điều kiện, 180 lượt MC, ba N0; lưu seed, photon và SHA-256 của sáu checkpoint cùng cấu hình. | Manifest kiểm chứng và hồ sơ phiên bản. | Danh sách và chỉ số đánh giá được lưu trước khi xem kết quả CNN. |
| V1 | Kiểm MC độc lập | Chạy pilot rồi hoàn thành ma trận MC trực tiếp. Không nội suy biên dạng từ LUT cũ để thay cho các lượt này. | Bản đồ R_xy, N0, log, seed, thống kê năng lượng và báo cáo chất lượng. | Có dữ liệu hợp lệ theo tiêu chí số; ghi rõ mọi điều kiện chưa hội tụ. |
| V2 | Trích đầu vào và dự đoán | Dùng tiền xử lý đã khóa; cho sáu CNN cùng dự đoán trên chính các biên dạng mới. Không cấp μa thật cho mạng. | Biên dạng 128 điểm và bảng dự đoán có mã điều kiện, mã MC lặp, seed mô hình, nhãn thật, F/K và sai số. | Truy ngược được từng dự đoán tới bản đồ MC; hai nhánh có cùng đầu vào và cùng nhãn. |
| V3 | Kiểm khái quát mức hấp thụ | Tách nhóm μa = 0,22 / 0,40 khỏi nhóm μa = 0,15 / 0,30 khi tổng hợp. Giữ cả mức chuẩn và cao trong báo cáo. | MAE, RMSE, bias và chênh lệch F/K theo nhóm. | Báo được F có giữ lợi ích ở mức chưa học hay chỉ ở các mức từng xuất hiện khi huấn luyện. |
| V4 | Giữ độ sâu, thay hấp thụ | Với từng z, vẽ dự đoán theo μa cho từng seed mô hình; kèm đường độ sâu thật. Tính khoảng biến thiên dự đoán và sai số. | Đồ thị dự đoán–μa, bảng độ nhạy theo từng độ sâu. | Kiểm đồng thời mức ổn định theo μa và mức gần nhãn thật; không gọi dự đoán hằng là khử được ảnh hưởng hấp thụ. |
| V5 | Giữ hấp thụ, thay độ sâu | Với từng μa, vẽ dự đoán theo độ sâu thật, đường y = x và sai số. So với cách luôn đoán độ sâu trung bình của tập train gốc. | Đồ thị dự đoán–độ sâu, MAE theo μa; hệ số góc và hệ số chặn mô tả. | CNN vẫn phân biệt được độ sâu; báo rõ vùng nén dải dự đoán hoặc sai lệch có hệ thống. |
| V6 | Tổng hợp đánh đổi và giới hạn | So sai số ghép cặp, báo từng seed và các vùng không đạt mục tiêu. Tách kết quả mức đã học, chưa học và chuẩn. | Bảng kết quả chính, bản đồ sai số theo z–μa và kết luận có điều kiện. | Kết luận không chỉ dựa trên một seed hoặc MAE gộp toàn bộ. |
| V7 | Chuẩn bị hệ thật | Kiểm kê thiết bị; làm mẫu DEV; đo mm/pixel, nhiễu, độ nhòe, đáp ứng camera và khả năng định lượng dung dịch tại 850 nm. | Hồ sơ hiệu chuẩn, công thức và quy trình DEV đã ổn định. | Xác định được phạm vi thông số thật và cách chuẩn hóa ảnh; ghi giới hạn nếu chưa đo được μa. |
| V8 | Kiểm chứng phantom | Theo quy trình đã có: năm độ sâu, P1/P2 trên cùng kênh, N0, ba phiên đặt lại × 20 ảnh; đo hình học thật sau chụp. Dùng mô hình đã khóa. | Ảnh gốc, metadata, ảnh mặt cắt, nhãn thật và dự đoán ghép cặp F/K. | Đánh giá được việc chuyển từ mô phỏng sang ảnh thật; ảnh lặp không được tính thành mẫu vật lý độc lập. |

Thứ tự: **V0 → V1 → V2 → V3/V4/V5 → V6**, sau đó kiểm chứng thực tế bằng **V7 → V8**. V7 có thể chuẩn bị song song khi điều kiện thiết bị cho phép. V0–V6 có thể thực hiện trước khi có dữ liệu thực nghiệm.

## 5 Chỉ số và cách tổng hợp phải dùng

| Chỉ số hoặc nhóm | Cách tính và cách diễn giải |
| --- | --- |
| Sai số có dấu | e = độ sâu dự đoán − độ sâu thật; dương là dự đoán sâu hơn, âm là nông hơn. |
| MAE | Trung bình của \|e\|. Tính riêng cho F/K, từng seed, từng μa và các nhóm đã quy định. |
| RMSE và bias | RMSE = căn trung bình e²; bias = trung bình e. Dùng để bổ sung cho MAE, không thay chỉ số chính sau khi xem kết quả. |
| Chỉ số chính mới | ΔMAE trên nhóm **chưa học μa = 0,22 và 0,40 mm⁻¹**, gồm 20 điều kiện z–μa: ΔMAE = MAE_K − MAE_F. Giá trị dương biểu thị F giảm MAE. |
| Nhóm yếu đã học | Báo riêng μa = 0,15 và 0,30 mm⁻¹, cùng nhóm gộp của hai mức. Đây là kiểm trên MC mới, không phải mức hấp thụ mới với F. |
| Đánh đổi ở mức chuẩn | Δchuẩn = MAE_F − MAE_K tại μa = 0,508 mm⁻¹. Báo cả số tuyệt đối và mức sai số thực tế; không bỏ qua khi F thua K. |
| Ổn định theo hấp thụ | Tại mỗi z và seed mô hình, lấy trung bình dự đoán qua ba MC lặp tại mỗi μa; tính R = maxμa(dự đoán) − minμa(dự đoán). Báo R_F/R_K cùng MAE tại z đó. R nhỏ chỉ có ý nghĩa khi dự đoán cũng gần độ sâu thật. |
| Khả năng phân biệt độ sâu | Tại mỗi μa, báo đường dự đoán–nhãn, MAE, hệ số góc và hệ số chặn. Hệ số góc gần 1 và chặn gần 0 là hướng mong muốn; không dùng riêng R² để kết luận chính xác. |
| Mục tiêu 0,5 mm | Báo miền điều kiện có MAE ≤ 0,5 mm và các điều kiện chưa đạt. Đây là mục tiêu đề tài, không phải chuẩn lâm sàng hoặc bảo đảm mọi dự đoán đều sai dưới 0,5 mm. |
| Tổng hợp lặp | Với mỗi điều kiện và seed mô hình, tính chỉ số qua ba MC lặp; sau đó tổng hợp đều trọng số theo điều kiện. Báo riêng từng seed mô hình, trung bình và độ lệch chuẩn giữa ba seed. |
| Tính ghép cặp | Tính chênh lệch sai số tuyệt đối K−F trên cùng điều kiện, cùng MC lặp và cặp seed mô hình; không so hai tập đầu vào khác nhau. |
| Tính độc lập | 60 điều kiện là lưới thiết kế cố định. Ba MC lặp đo biến thiên số học; ba seed CNN đo biến thiên huấn luyện. Không nhân các số này để tuyên bố cỡ mẫu vật lý lớn hơn. Ba N0 dùng chung theo chỉ số lặp cũng tạo liên hệ giữa các điều kiện. |
| Suy luận thống kê | Giai đoạn này ưu tiên mô tả ghép cặp, độ lớn hiệu ứng và tính nhất quán theo seed/điều kiện. Không tuyên bố ý nghĩa thống kê trên quần thể chỉ từ ba seed hoặc từ cách đếm lặp thành mẫu độc lập. |

## 6 Tiêu chí kết luận

| Kết quả quan sát | Kết luận được phép |
| --- | --- |
| F giảm MAE trên MC mới ở μa = 0,15 / 0,30 | Lợi ích trên các mức hấp thụ F đã học giữ được khi đổi lượt MC và độ sâu kiểm. Chưa đủ để nói khái quát sang μa mới. |
| F giảm MAE nhóm 0,22 / 0,40 ở cả ba seed, đồng thời vẫn phân biệt được độ sâu | Bằng chứng nhất quán cho khả năng khái quát tới các mức hấp thụ trung gian chưa huấn luyện, trong mô hình vật lý đã khảo sát. |
| ΔMAE trung bình dương nhưng một seed không cải thiện | Báo lợi ích trung bình kèm sự thiếu nhất quán; không diễn đạt thành cải thiện ổn định ở mọi lần huấn luyện. |
| R_F nhỏ hơn R_K và MAE_F cũng thấp hơn | Hỗ trợ nhận định F ít bị ảnh hưởng bởi hấp thụ hơn K ở các độ sâu đó. Không suy ra F đã nhận diện duy nhất cả μa và độ sâu. |
| R_F nhỏ nhưng dự đoán gần hằng hoặc sai độ sâu lớn | Chưa có bằng chứng tách ảnh hưởng hấp thụ khỏi độ sâu một cách hữu ích. |
| F kém K tại mức chuẩn | Ghi rõ đánh đổi; kết luận F có lợi trong miền nào thay vì tuyên bố F tốt hơn toàn bộ. |
| Có vùng MAE > 0,5 mm | Khoanh vùng giới hạn; không dùng MAE gộp thấp để che các điều kiện không đạt. |
| F không cải thiện trên bộ mới | Giả thuyết chưa được hỗ trợ trong phép kiểm này; giữ và báo kết quả. Nếu sửa mô hình, mở giai đoạn phát triển mới và dành một bộ kiểm chứng khác. |
| F có lợi trên phantom độc lập với quy trình chụp đã khóa | Bổ sung bằng chứng chuyển sang ảnh thật trên các khối, đường kính, vật liệu và bố trí đã kiểm; chưa suy rộng sang người hoặc mọi loại tĩnh mạch. |

Không đặt điều kiện hoàn thành là phải thu được kết quả thuận lợi cho F. Hoàn thành là chạy đúng phép kiểm đã khóa, báo đầy đủ kết quả và giới hạn, kể cả khi giả thuyết không được hỗ trợ.

## 7 Quy định riêng cho kiểm chứng phantom

Giữ năm độ sâu mép trên dự kiến 0,75 / 1,25 / 2 / 3 / 4,5 mm; D danh nghĩa 3 mm; thêm N0 và mẫu phát triển DEV riêng. P1/P2 là hai điều kiện dung dịch trên cùng kênh, không phải tên hai mô hình F/K. Mỗi ảnh kiểm chứng được cả F và K dự đoán.

P1/P2 có μa mục tiêu 0,508 / 0,30 mm⁻¹. Nếu chỉ biết nồng độ hoặc đo suy hao tổng, báo đúng đại lượng đó; không gán thành μa tinh khiết. Chụp đủ dữ liệu trước khi cắt; nhãn đánh giá lấy từ hình học đo thật, không lấy cao độ gá thay số đo cuối.

Năm khối × hai dung dịch × ba phiên × 20 khung = 600 ảnh kỹ thuật, không phải 600 mẫu vật lý độc lập. Tổng hợp theo khối, điều kiện và phiên; giữ ảnh gốc và lý do loại lỗi kỹ thuật. Không loại ảnh vì mô hình dự đoán sai.

Nếu hiệu chuẩn camera hoặc mẫu DEV cho thấy cần cập nhật dữ liệu mô phỏng và huấn luyện lại, lập phiên bản mô hình mới trước khi mở dữ liệu của năm khối cuối. Báo riêng kết quả sáu checkpoint cũ và phiên bản mới nếu cùng được đánh giá; không chọn phiên bản dựa trên kết quả phantom cuối rồi gọi đó là kiểm chứng độc lập.

## 8 Hồ sơ bàn giao và các mốc

Đường dẫn dưới đây là vị trí **dự kiến**, chưa phải các thư mục kết quả đã được tạo.

| Mốc | Sản phẩm | Vị trí dự kiến | Cần dữ liệu thật? |
| --- | --- | --- | --- |
| M0 Khóa phép kiểm | Manifest, seed, hash mô hình/cấu hình và chỉ số định trước. | `configs/validation_independent.json` | Không |
| M1 MC độc lập | 180 bản đồ có kênh, ba N0 nếu không cần lượt bổ sung; báo cáo chất lượng và các lượt không đạt. | `exp1_homogeneous/06_final_test/results/mc/` | Không |
| M2 Dự đoán | Biên dạng 128 điểm, nhãn và bảng dự đoán sáu mô hình trên cùng đầu vào. | `exp1_homogeneous/06_final_test/results/predictions/` | Không |
| M3 Kết luận trong mô phỏng | Bảng chỉ số đã học/chưa học/chuẩn/cao; biểu đồ theo μa và z; miền đạt mục tiêu; giới hạn. | `exp1_homogeneous/06_final_test/results/report/` | Không |
| M4 Hệ thật sẵn sàng | Hiệu chuẩn camera, hồ sơ DEV, dung dịch và quy trình chụp. | `results/phantom_development/` | Có |
| M5 Phantom độc lập | Ảnh gốc, số đo thật, kết quả F/K và kết luận chuyển sang thực nghiệm. | `results/phantom_validation/` | Có |

Ưu tiên bắt đầu bằng **V0 và pilot của V1**. Sau khi xác nhận chất lượng MC, hoàn thành ma trận, chạy sáu checkpoint đã khóa và tổng hợp V3–V6. Không cần tạo thêm hàng nghìn cảnh nội suy để thay cho phép kiểm độc lập này.

## 9 Tài liệu và kết quả liên quan

- [Kế hoạch bốn bước và kết quả hiện có](Ke_hoach_4_buoc_mo_phong_CNN_FK.md).
- [Hướng dẫn chạy chương trình và giới hạn mô phỏng](README.md).
- Cấu hình hiện tại: `configs/simulation.json`, `configs/learning.json`.
- Kết quả đã xem: `exp1_homogeneous/04_training/results/FK/comparison.json`, `exp1_homogeneous/04_training/results/FK/comparison.csv`, `exp1_homogeneous/maintenance/audit.json`.
- Quy trình thực nghiệm: `Huong_dan_chi_tiet_thuc_hien_phantom_phan_xa.docx`.
- Vật tư và thiết bị: `Danh_muc_thiet_bi_vat_tu_dung_cu_phantom_phan_xa.docx`.
