# Chạy mô phỏng phản xạ và CNN F K

Đây là bộ chạy mới theo kế hoạch một kênh 3 mm, nền cố định và g = 0,511. Các kết quả MC ở thư mục cha có cấu hình khác và không được dùng thay thế cho kết quả mới.

## Môi trường đã kiểm tra

- GPU: AMD Radeon RX 6600, OpenCL nhận diện thành công.
- Mô phỏng: `E:/thacsi/LV/vein/new_pp/.venv-fk/Scripts/python.exe`, pmcxcl 0.7.5.
- Phân tích và CNN: Python 3.12.8 hiện có, NumPy, SciPy, Matplotlib và PyTorch CPU.
- Cấu hình nghiên cứu: `configs/simulation.json`.

Các lệnh dưới đây chạy trong thư mục dự án bằng PowerShell. Không cần cài hoặc thay đổi môi trường cũ.

```powershell
python -m unittest discover -s code -p 'test_*.py'
& 'E:/thacsi/LV/vein/new_pp/.venv-fk/Scripts/python.exe' -u code/stage1.py > stage1_run.log 2>&1
python code/report_stage1.py
& 'E:/thacsi/LV/vein/new_pp/.venv-fk/Scripts/python.exe' -u code/stage2.py > stage2_run.log 2>&1
```

Chạy tuần tự toàn kế hoạch, dừng nếu có lỗi hoặc kiểm số học không đạt:

```powershell
./run_plan.ps1
```

Chạy tiếp một khoảng giai đoạn đã đủ điều kiện, ví dụ bước 2–4:

```powershell
./run_plan.ps1 -FromStage 2 -ToStage 4
```

Không khởi động hai bản chạy trên GPU cùng lúc. Xem `exp1_homogeneous/maintenance/plan_status.json` để biết giai đoạn của bộ chạy tuần tự. Các chương trình đã chuẩn bị cho giai đoạn sau không có nghĩa giai đoạn đó đã chạy xong; chỉ xác nhận bằng báo cáo và đầu ra thực tế.

Kiểm tra độc lập các tệp kết quả đã hoàn tất:

```powershell
python code/audit_results.py
```

Các đầu ra chính của lần chạy đã hoàn tất gồm `exp1_homogeneous/04_training/results/FK/comparison.json`, `comparison.csv`, `comparison.png`, `mae_depth_absorption.png`, dữ liệu trong `exp1_homogeneous/03_dataset/data/dataset_FK` và sáu checkpoint trong `exp1_homogeneous/04_training/results/FK/models`. Kế hoạch Markdown trong thư mục gốc đã được cập nhật với kết quả và các giới hạn.

## Dữ liệu và học máy

Bộ thử CNN có 1.200 cảnh (960/120/120); bộ F/K chính thức có 6.000 cảnh mỗi nhánh (4.800/600/600). Độ sâu lấy đều trong 0,5–5 mm; vị trí ngang lệch trong ±2 mm, ghép cặp giữa F/K. Quan sát lý tưởng không thêm nhiễu camera hoặc độ nhòe; vẫn còn nhiễu ước lượng Monte Carlo trong bảng tra.

Các cảnh được sinh bằng nội suy cùng bảng tra MC. Chia theo mã cảnh ngăn các biến thể cùng cảnh sang nhiều tập, nhưng không biến test nội suy thành phép thử MC độc lập hoặc thử nghiệm vật lý. Các lượt MC ở độ sâu trung gian chỉ kiểm nội suy và được lưu riêng, không thêm làm nút bảng tra.

CNN dùng ba tích chập 16/32/64 kênh, kernel 5, padding 2, stride 1, gộp trung bình toàn cục, lớp ẩn 32 và đầu ra một độ sâu. Quy tắc học cố định ở `configs/learning.json`. Bước 4 huấn luyện F/K với ba seed và chấm cả hai nhánh trên cùng 600 đầu vào test F; nhóm hấp thụ yếu có 300 mẫu là phép so sánh chính.

Mỗi lượt mô phỏng lưu NPZ và JSON chứa cấu hình, seed, số photon, phiên bản công cụ, kết quả kiểm hình học và thống kê từng batch. Có thể chạy lại sau gián đoạn: chỉ bỏ qua tệp đã hoàn tất nếu định danh cấu hình, seed, số photon và phiên bản pmcxcl khớp. Không ghi đè tệp của cấu hình khác.

Lượt kiểm lưới 0,05 mm dùng batch 1 triệu photon vì batch 6 triệu trả NaN trên môi trường AMD/Windows hiện có. Tổng ngân sách photon không giảm. Dữ liệu lỗi được lưu riêng trong `exp1_homogeneous/maintenance/diagnostics`, không được dùng tạo bảng tra hoặc huấn luyện.

## Cách đọc kết quả bước 1

- `exp1_homogeneous/01_simulator_checks/results/status.json`: công việc đang chạy và danh sách đã hoàn tất.
- `exp1_homogeneous/01_simulator_checks/results/report.json`: chỉ được tạo khi phân tích; xem `gate_pass`, từng phép kiểm và danh sách thiếu.
- `exp1_homogeneous/01_simulator_checks/results/stage1_overview.png`: hình học, bản đồ phản xạ và các biên dạng.
- `exp1_homogeneous/01_simulator_checks/results/input_128.npz`: một đầu vào 128 điểm ở độ sâu 2 mm, chưa đủ để huấn luyện CNN đo nhiều độ sâu.

## Quy ước vật lý

MC thu `dref` ở lớp không khí phía trên khối. Đây là tổng phản xạ thoát ra theo các hướng, chưa mô hình hóa khẩu độ thu, kính lọc, LED vòng hoặc cảm biến thật. Không dùng fluence trong thể tích để thay cho phản xạ bề mặt.

Khi so tổng phản xạ giữa các lưới, tích phân mật độ `dref` trên diện tích: `sum(R_xy) * voxel_mm**2`. Không so trực tiếp tổng các phần tử khi kích thước voxel khác nhau. Báo cáo bước 1 lưu giá trị tích phân và kiểm giới hạn năng lượng.

Miền ngang tuần hoàn; trục y dài 8 mm và có cùng tiết diện kênh ở mọi y. Điều kiện này mô tả kênh thẳng vô hạn theo trục y, không phải hai đầu kênh trong phantom hữu hạn. Bước kiểm miền ngang tăng x từ 60 lên 80 mm để đánh giá tương tác giữa các kênh ảnh tuần hoàn theo x.

Biên dạng chia N0 tại từng điểm x-y trước, lấy trung bình dọc y, sau đó tích phân trung bình theo 128 ô ngang trong vùng 32 mm. Dữ liệu đầu vào lưu nguyên tỉ số, không min–max từng mẫu. Làm trơn Gaussian và đối xứng chỉ dùng chẩn đoán tương phản/độ rộng, không tự động áp dụng cho đầu vào CNN. Phiên bản phép trích được lưu trong báo cáo và bảng tra; nếu phiên bản đổi, tính lại từ bản đồ MC gốc trước khi tạo dữ liệu học.

Ngưỡng kiểm tương phản và độ rộng là 10%. Nếu tương phản tham chiếu nhỏ hơn 0,05 thì dùng thêm ngưỡng chênh lệch tương phản tuyệt đối 0,005; độ rộng vẫn phải xác định được và đạt kiểm tương đối. Đây là tiêu chí số học của dự án, không phải độ chính xác thực nghiệm.

## Tài liệu công cụ

- [MCX-CL Python bindings](https://github.com/fangq/mcxcl/tree/master/pmcxcl).
- [MCX-CL cấu hình và đại lượng đầu ra](https://github.com/fangq/mcxcl/blob/master/mcxlabcl/mcxlabcl.m).

Các bảng tra mô phỏng và giả định quan sát phải được ghi rõ trong mọi kết luận. Chưa có hiệu chuẩn camera hoặc kiểm chứng phantom trong bộ chạy này.
