# nir-vein-depth

Ước lượng độ sâu tĩnh mạch nông từ **một ảnh phản xạ khuếch tán 850 nm**. Phương pháp gồm mô phỏng Monte Carlo trên GPU (pmcxcl) và mạng tích chập huấn luyện trên profile mạch mô phỏng.

## Kết quả chính

Bộ test khóa gồm 40 cấu hình da nhiều lớp chưa từng dùng (120 mạch), mở đúng một lần theo kế hoạch ghi trước (`exp2_layered/06_final_test/report_locked_test.md`). Mô hình cuối là hetero WideCNN (ensemble 3 seed):

- **MAE 0,206 mm** (KTC 95 % 0,179–0,235 mm, bootstrap theo làn da), trung bình ba mức nhiễu camera.
- Độ bất định ±1,96σ phủ 92–96 % sai số.

So sánh trên cùng bộ test:

| Mô hình | MAE (mm) |
|---|---|
| hetero (mô hình cuối) | 0,206 |
| wide | 0,207 |
| F_layered | 0,224 |
| CNN da đồng nhất (thử nghiệm 1) | 1,232 |
| B3 khớp vật lý đồng nhất | 1,270 |
| Đoán hằng số | 0,859 |

Tất cả kết quả trên là **mô phỏng**. Phantom mới ở mức kế hoạch, chưa thực hiện. Ảnh tay thật không có độ sâu chuẩn.

## Cấu trúc

Hai thử nghiệm dùng **chung một khung thư mục**; cùng số thì cùng vai trò:

| Thư mục bước | Vai trò | Thử nghiệm 1 (da đồng nhất) | Thử nghiệm 2 (da nhiều lớp) |
|---|---|---|---|
| `common/` | code dùng chung | lõi MC đã khóa: `simulate.py`, `readout.py`, `train.py` | `cnn_variants.py`, `b3_physics_fit.py` |
| `01_simulator_checks/` | MC đã hội tụ, chạy đúng chưa | số photon, lưới, miền, thời gian, chạy lặp | khả năng pmcxcl, MC CPU so với GPU |
| `02_forward_model/` | profile phụ thuộc độ sâu (và các yếu tố khác) thế nào | quét độ sâu ở hấp thụ chuẩn | giới hạn thông tin (CRLB), mô hình da nhiều lớp |
| `03_dataset/` | dữ liệu mô phỏng (`data/`) | 48 lần MC → bảng tra → 6.000 cảnh nội suy cho mỗi nhánh F, K | MC 300 da train, 30 validation, 90 validation_ext, 40 locked_test |
| `04_training/` | huấn luyện, so sánh mạng | CNN thử K; CNN F so với K, 3 seed | CNN da nhiều lớp, các biến thể, **mô hình cuối** hetero |
| `05_error_analysis/` | vì sao còn sai số | — | giới hạn 14 tham số, sai lệch cố định, huấn luyện lại |
| `06_final_test/` | kiểm tra cuối trên dữ liệu chưa từng dùng | kiểm chứng độc lập bằng MC trực tiếp (V0–V6) | bộ test khóa, mở một lần |
| `maintenance/` | vận hành: chuyển máy, so phần cứng, audit | bản lưu máy AMD | — |

Mỗi bước có `code/`, kết quả (`results/` hoặc `data/`) và báo cáo. Mỗi thử nghiệm có `README.md` giải thích file dữ liệu, mô hình và các từ viết tắt.

| Thư mục khác | Nội dung |
|---|---|
| `real_hand/` | Ảnh tay thật. **Chỉ có code, không kèm ảnh** |
| `phantom/` | Kế hoạch phantom và phép tính thiết kế |
| `colab/` | Notebook chạy kiểm tra và đánh giá, mỗi ô một việc |
| `tools/` | `compare_regenerated.py`: so dữ liệu MC tạo lại với dữ liệu trong repo |
| `docs/` | Danh sách hash các file khóa |
| `requirements/` | Môi trường: `mc.txt` (GPU) và `analysis.txt` (CPU) |

Cách cài đặt, cách chạy lại và các quy ước về file khóa, bộ test khóa, dữ liệu tay thật xem trong [README.md](README.md).

Muốn tự chạy lại mô phỏng để tạo dữ liệu (cần GPU, mất nhiều giờ) và kiểm kết quả: xem **[REPRODUCE_vi.md](REPRODUCE_vi.md)**.
