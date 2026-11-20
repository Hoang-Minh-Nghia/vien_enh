# Hướng dẫn chuyển dự án sang máy RTX 3060 và chạy tiếp V1

Làm lần lượt từ trên xuống. Các lệnh chạy trong **PowerShell**.

## 1 Tình trạng dự án khi đóng gói

| Nội dung | Trạng thái |
| --- | --- |
| Giai đoạn 1–4 (mô phỏng, dữ liệu, CNN, so sánh F/K) | Đã xong, 6 checkpoint đi kèm trong gói |
| V0 khóa phép kiểm | Đã xong: `configs/validation_independent.json` |
| V1 pilot trên AMD RX 6600 | Dừng giữa chừng: 3 N0 và 9/18 lượt có kênh; đã **lưu trữ riêng, không dùng tiếp** |
| V1 pilot và ma trận trên RTX 3060 | **Việc cần làm tiếp**, chạy từ đầu theo hướng dẫn này |
| V2 trở đi (dự đoán CNN trên dữ liệu mới) | Chưa làm; chưa mở dự đoán nào |

**Gói này khác bản gốc ở hai điểm:**

1. Thư mục kết quả AMD được đổi tên thành `exp1_homogeneous/maintenance/archive_AMD_RX6600`. Thư mục `exp1_homogeneous/06_final_test/results` chưa tồn tại, nên pilot sẽ chạy mới hoàn toàn trên RTX.
2. Manifest có thêm một mục trong `amendments` ghi việc đổi phần cứng, và SHA-256 trong `configs/validation_independent.sha256` đã cập nhật tương ứng. Thiết kế, seed, số photon và chỉ số đánh giá **không đổi**.

**Vì sao không chạy tiếp trên phần dữ liệu AMD:** mỗi lượt có kênh được chia cho N0 cùng lượt lặp. Nếu N0 từ AMD còn lượt có kênh từ RTX thì mọi sai khác giữa hai GPU sẽ đi thẳng vào tương phản. Chạy lại chỉ tốn khoảng 20 phút.

## 2 Yêu cầu máy mới

| Thành phần | Yêu cầu |
| --- | --- |
| Hệ điều hành | Windows 10/11 64-bit |
| GPU | NVIDIA RTX 3060 với driver NVIDIA mới (có sẵn OpenCL) |
| Python | **3.12** bản 64-bit, từ python.org; chọn cài `py launcher` |
| Ổ đĩa | Khoảng 2 GB trống là đủ cho cả hai môi trường và kết quả tier 0 |
| Internet | Cần để cài thư viện, trừ khi đã có sẵn gói |

Trong lúc chạy MC, không chạy game, phần mềm dựng hình hay tác vụ GPU nặng khác.

## 3 Giải nén

Giải nén `FK_RaspberryPi_RTX3060.zip` vào một đường dẫn ngắn, **không dấu tiếng Việt, không khoảng trắng**, ví dụ `D:\fk`. Bạn sẽ có `D:\fk\huong_phan_xa_FK_RaspberryPi`.

Không mở rồi lưu lại các file trong `code/` và `configs/`, và không dùng Git với chế độ tự đổi xuống dòng. Khóa V0 kiểm tra SHA-256 từng byte, nên đổi một ký tự cũng làm bộ chạy từ chối.

```powershell
cd D:\fk\huong_phan_xa_FK_RaspberryPi
```

Mọi lệnh sau đều chạy trong thư mục này.

## 4 Cài hai môi trường Python

Dự án dùng hai môi trường tách riêng, giống máy cũ: một cho mô phỏng, một cho phân tích và CNN.

```powershell
py -3.12 -m venv ..\.venv-fk
..\.venv-fk\Scripts\python.exe -m pip install -r env\requirements_mc.txt
```

```powershell
py -3.12 -m venv ..\.venv-analysis
..\.venv-analysis\Scripts\python.exe -m pip install -r env\requirements_analysis.txt
```

**Nếu pip báo không tìm thấy `pmcxcl==0.7.5`:** không cài bản khác, vì khóa V0 yêu cầu đúng bản này. Dùng bản dự phòng trong gói:

```powershell
..\.venv-fk\Scripts\python.exe -m pip install -r env\requirements_mc_no_pmcxcl.txt
Copy-Item env\pmcxcl_0.7.5_offline\* ..\.venv-fk\Lib\site-packages -Recurse -Force
```

Không dùng `run_plan.ps1` trên máy mới vì file đó ghi cứng đường dẫn máy cũ.

## 5 Kiểm tra máy mới

Chạy kiểm tra chương trình và dữ liệu:

```powershell
..\.venv-analysis\Scripts\python.exe -m unittest discover -s code -p 'test_*.py'
```

Kết quả mong đợi: `Ran 17 tests` và `OK`.

Chạy kiểm tra máy. Bước này chạy thử khoảng 1 phút trên GPU và không đụng vào dữ liệu kiểm chứng:

```powershell
..\.venv-fk\Scripts\python.exe -u code\check_new_machine.py
```

Dòng cuối phải là `MACHINE_CHECK_PASS`. Kết quả lưu ở `exp1_homogeneous/maintenance/migration_check/machine_check.json`, gồm tên GPU đã dùng và tốc độ thực tế (`photons_per_second`, `estimated_matrix_hours_tier0`). So với mức khoảng 1,2 triệu photon/giây của RX 6600 để biết RTX nhanh hơn bao nhiêu.

Dòng `please first install pmcx module...` là cảnh báo vô hại, không cần xử lý.

## 6 Chạy pilot V1

Khoảng 20 phút trên RX 6600, trên RTX có thể khác.

```powershell
..\.venv-fk\Scripts\python.exe -u code\v1_run_mc.py --pilot --max-tier 0 *> v1_pilot_run_RTX3060.log
```

Xem tiến độ ở cửa sổ PowerShell khác:

```powershell
Get-Content v1_pilot_run_RTX3060.log -Tail 5 -Wait
```

Sau khi xong, mở `exp1_homogeneous/06_final_test/results/quality_pilot.json` và xem `gate_pass`, `n_accepted`.

So sánh RTX với AMD (dùng 9 lượt pilot AMD đã lưu):

```powershell
..\.venv-fk\Scripts\python.exe code\compare_hardware.py
```

Xem `HARDWARE_COMPARISON pass=...`. Báo cáo ở `exp1_homogeneous/06_final_test/results/hardware_comparison.json`.

| Kết quả | Việc làm tiếp |
| --- | --- |
| `gate_pass` đúng và so phần cứng `pass=True` | Sang mục 7 |
| `gate_pass` sai vì chưa hội tụ | Tăng photon theo mức đã khóa: chạy lại lệnh pilot với `--max-tier 1`, nếu vẫn chưa đạt thì `--max-tier 2` |
| `pass=False` khi so phần cứng | **Dừng**, chưa chạy ma trận, chưa làm V2. Hai GPU cho kết quả khác nhau vượt ngưỡng, cần xem xét vì CNN được huấn luyện từ dữ liệu AMD |

## 7 Chạy toàn bộ ma trận V1

Chỉ làm sau khi mục 6 đạt. Thời gian ước tính nằm trong `machine_check.json`; trên RX 6600 khoảng 2,5–3 giờ, chưa kể các điều kiện phải tăng photon.

```powershell
..\.venv-fk\Scripts\python.exe -u code\v1_run_mc.py --max-tier 2 *> v1_matrix_run_RTX3060.log
```

Các lượt pilot đã đạt được bỏ qua tự động. Báo cáo cuối: `exp1_homogeneous/06_final_test/results/quality_matrix.json`.

Với `--max-tier 2`, điều kiện chưa đạt sẽ tự chạy lại ở 1,2 × 10⁸ rồi 2,4 × 10⁸ photon, nên trường hợp xấu nhất có thể lâu hơn nhiều. Điều kiện vẫn chưa đạt ở mức cao nhất được ghi `accepted = false`; đó là kết quả cần báo cáo, không phải lỗi cần che.

**Dừng và chạy tiếp:** bấm `Ctrl+C` hoặc tắt máy đều được. Chạy lại đúng lệnh cũ, các lượt đã xong được bỏ qua; lượt đang dở sẽ chạy lại từ đầu với cùng seed.

## 8 Lỗi thường gặp

| Hiện tượng | Nguyên nhân và cách xử lý |
| --- | --- |
| `Locked file changed` hoặc `Manifest does not match` | File bị sửa hoặc đổi xuống dòng. Giải nén lại từ zip gốc, đừng sửa hash cho khớp |
| `selected_gpu: false` | Thiết bị OpenCL số 1 không phải RTX, thường do có GPU tích hợp Intel/AMD. Xem `opencl_devices` trong `machine_check.json`, rồi tắt GPU tích hợp trong Device Manager hoặc BIOS. **Không sửa `gpuid`** trong `simulation.json` vì tệp đó đã bị khóa |
| Không có thiết bị OpenCL nào | Cài lại driver NVIDIA bản đầy đủ từ trang chủ NVIDIA |
| `Invalid top reflectance`, NaN | Lỗi số trên GPU. Lưu lại `.invalid.npz`, dừng chạy, không xóa hay sửa điểm lỗi rồi chạy tiếp |
| `Refusing to overwrite a different run` | Tệp cũ có cấu hình khác. Không xóa vội; kiểm tra xem cấu hình hoặc phiên bản `pmcxcl` có bị đổi không |
| `DLL load failed` khi nhập `pmcxcl` | Cài Microsoft Visual C++ Redistributable 2015–2022 x64 và cập nhật driver NVIDIA |

## 9 Những việc không được làm

- Không đổi `configs/simulation.json`, `learning.json`, `code/simulate.py`, `readout.py`, `train.py`.
- Không dùng `pmcx` (bản CUDA) thay `pmcxcl` trong đợt kiểm chứng này. Đó là công cụ khác và chưa được khóa.
- Không chạy dự đoán CNN trên dữ liệu mới trước khi V1 hoàn tất và lựa chọn dữ liệu đã chốt theo chất lượng MC.
- Không chạy hai bộ chạy GPU cùng lúc.
- Không xóa thư mục lưu trữ AMD, vì `compare_hardware.py` cần nó.
- Không loại điều kiện vì CNN dự đoán sai, và không thử thêm seed để giữ lượt thuận lợi.

## 10 Sau khi xong V1

Giữ nguyên các file sau để làm tiếp V2 (ở máy mới hoặc mang về máy cũ):

- `exp1_homogeneous/06_final_test/results/` (toàn bộ)
- `exp1_homogeneous/maintenance/migration_check/machine_check.json`
- `v1_pilot_run_RTX3060.log`, `v1_matrix_run_RTX3060.log`

Nếu dùng Claude Code trên máy mới, mở đúng thư mục dự án và yêu cầu làm V2. Các tài liệu cần đọc: [kế hoạch kiểm chứng](Ke_hoach_tiep_theo_kiem_chung_doc_lap_FK.md) và [README](README.md).
