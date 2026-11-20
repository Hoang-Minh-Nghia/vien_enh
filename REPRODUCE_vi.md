# Tạo lại dữ liệu mô phỏng

Repo đã có sẵn toàn bộ dữ liệu mô phỏng và mô hình, nên các phép đánh giá chạy được ngay trên CPU. Hướng dẫn này dành cho người muốn **tự chạy lại mô phỏng Monte Carlo (MC) từ đầu** trên máy mình, rồi so kết quả với dữ liệu trong repo.

> English: [REPRODUCE.md](REPRODUCE.md)

## Nguyên tắc

- Mọi script sinh dữ liệu **bỏ qua file đã có**. Bị ngắt giữa chừng thì chạy lại đúng lệnh đó, script sẽ làm tiếp. Chạy ngay trong repo vừa clone thì không có gì xảy ra, vì dữ liệu đã đủ.
- Muốn tạo lại thì dùng **một bản clone riêng**: xóa các thư mục dữ liệu cần làm lại, chạy các bước bên dưới, rồi so với bản gốc bằng `tools/compare_regenerated.py`.
- File cấu hình được sinh từ seed cố định trước khi chạy, nên lần chạy mới mô phỏng đúng những làn da, mạch và số photon như cũ.
- Mỗi lúc **chỉ chạy một tiến trình MC**: mọi script đều dùng GPU thứ nhất (`gpuid = 1`).

## 1. Chuẩn bị

| | Đã dùng cho repo này |
|---|---|
| Hệ điều hành | Windows 11 (các lệnh dưới là PowerShell) |
| GPU | GPU bất kỳ có OpenCL. Thời gian đo: thử nghiệm 1 trên AMD Radeon RX 6600, thử nghiệm 2 trên NVIDIA RTX 3060 |
| Python | 3.12 |
| Ổ đĩa | trống khoảng 1 GB |

```powershell
git clone https://github.com/Hoang-Minh-Nghia/vien_enh.git vien_enh_repro
cd vien_enh_repro
python -m venv .venv-fk;       .\.venv-fk\Scripts\pip install -r requirements\mc.txt              # MC (pmcxcl, GPU)
python -m venv .venv-analysis; .\.venv-analysis\Scripts\pip install -r requirements\analysis.txt  # phân tích, huấn luyện (CPU)
.\.venv-fk\Scripts\python -c "import pmcxcl; print(pmcxcl.gpuinfo())"                             # phải thấy GPU trong danh sách
```

Mọi lệnh bên dưới chạy từ thư mục gốc `vien_enh_repro`.

## 2. Thử nghiệm 1 — da đồng nhất

**Xóa kết quả cần làm lại.** Giữ nguyên `common/` và `configs/` (đầu vào đã khóa).

```powershell
Remove-Item -Recurse -Force exp1_homogeneous\01_simulator_checks\results, exp1_homogeneous\02_forward_model\results, `
  exp1_homogeneous\03_dataset\data, exp1_homogeneous\04_training\results
```

**Chạy cả 4 bước.** Script dừng ngay nếu một phép kiểm số học không đạt.

```powershell
cd exp1_homogeneous
.\run_plan.ps1 -AnalysisPython ..\.venv-analysis\Scripts\python.exe     # phần MC mặc định dùng ..\.venv-fk
cd ..
```

Chi tiết từng bước của `run_plan.ps1`. Muốn chạy riêng một bước thì dùng `-FromStage`/`-ToStage`, ví dụ `.\run_plan.ps1 -FromStage 2 -ToStage 2`.

| Bước | Lệnh (trong `exp1_homogeneous/`) | Môi trường | Thời gian đo | Đầu ra |
|---|---|---|---|---|
| 1 | `01_simulator_checks/code/mc_convergence.py`, rồi `convergence_report.py` | GPU, rồi CPU | 26 phút | `01_simulator_checks/results/` (10 lần chạy MC, `report.json` có `gate_pass`) |
| 2 | `02_forward_model/code/depth_sweep.py` | GPU | 12 phút | `02_forward_model/results/` (12 độ sâu, `lut_K.npz`, `report.json`) |
| 3 | `03_dataset/code/lut_dataset.py`; `common/train.py --dataset 04_training/results/pilot_K/data/dataset.npz --output 04_training/results/pilot_K --branch K --seed 42 --test-input K`; `02_forward_model/code/plot_depth_profiles.py` | CPU | vài phút | bộ dữ liệu thử và CNN K thử trong `04_training/results/pilot_K/` |
| 4 | `03_dataset/code/mc_runs_FK.py`, rồi `04_training/code/train_FK.py` | GPU, rồi CPU | 34 phút MC + khoảng 3 phút huấn luyện | `03_dataset/data/` (41 lần chạy MC: 36 điểm bảng tra + 5 phép kiểm, `lut_FK.npz`, `dataset_FK/`); `04_training/results/FK/` (6 mô hình, `comparison.json`) |

Tiến độ được ghi vào `maintenance/plan_status.json`. Output của từng lệnh nằm trong các file `stage*.log` cạnh script.

**Kiểm chứng độc lập (bước 06, tùy chọn).** Chạy MC trực tiếp với seed mới và mức hấp thụ chưa huấn luyện: 162 lần chạy, khoảng 3 giờ trên RTX 3060.

```powershell
cd exp1_homogeneous
..\.venv-fk\Scripts\python 06_final_test\code\run_independent_mc.py --max-tier 2
..\.venv-analysis\Scripts\python 06_final_test\code\predict_independent.py
..\.venv-analysis\Scripts\python 06_final_test\code\report_independent.py
cd ..
```

Bước này được khóa bằng SHA-256 vào **đúng các mô hình F/K đang lưu** (xem `06_final_test/plan_independent_validation.md`). Vì vậy phải chạy nó trước khi tạo lại `04_training`, hoặc chạy trong một bản clone mà `04_training/results/FK/models` chưa bị thay. Nếu mô hình đã khác, script sẽ từ chối chạy.

## 3. Thử nghiệm 2 — da nhiều lớp

**Xóa kết quả cần làm lại.** Lệnh này xóa cả manifest, để manifest được sinh lại từ seed.

```powershell
Remove-Item -Recurse -Force exp2_layered\03_dataset\data
```

**Sinh dữ liệu** (trong `exp2_layered/03_dataset/code/`):

```powershell
cd exp2_layered\03_dataset\code
$MC = "..\..\..\.venv-fk\Scripts\python.exe"; $PY = "..\..\..\.venv-analysis\Scripts\python.exe"

& $MC dataset_generate.py manifest              # manifest.json: 300 da train, 30 validation, 40 locked_test
& $MC dataset_generate.py run                   # chạy train, validation, rồi locked_test           (~9 giờ)
& $MC dataset_extend_ext.py manifest            # manifest_ext.json
& $MC dataset_extend_ext.py run validation_ext  # 90 da                                             (~3,7 giờ)
& $PY dataset_check.py                          # kiểm nhanh train/validation (không mở locked_test)
& $PY dataset_summary.py                        # ghi data/check_train_validation.json
cd ..\..\..
```

| Lệnh | Đầu ra (trong `exp2_layered/03_dataset/data/`) | Thời gian đo (RTX 3060) |
|---|---|---|
| `dataset_generate.py run` | `train/` 300, `validation/` 30, `locked_test/` 40 file | 9,0 giờ |
| `dataset_extend_ext.py run validation_ext` | `validation_ext/` 90 file | 3,7 giờ |
| tùy chọn: `dataset_high_snr_a.py manifest`, rồi `dataset_high_snr_a.py run` | `train_hisnr/` 120 file (120 da train đầu, gấp 8 lần photon; chỉ dùng cho phép thử sàn nhiễu) | khoảng 18 giờ (ước tính) |

- **Bị ngắt:** chạy lại đúng lệnh đó. Làn da đã xong được bỏ qua, file đang ghi dở (`*.partial.npz`) được ghi lại.
- **Lỗi MC:** script ghi `<id>.invalid.json` rồi dừng. Giữ file đó và tìm nguyên nhân trước khi chạy tiếp; không xóa file để ép chạy lại.
- **locked_test:** bộ `locked_test/` tạo lại vẫn là bộ test cuối. Không dùng nó để chọn hay chỉnh mô hình.
- `dataset_extend_ext.py run` nếu không ghi tên bộ sẽ chạy tiếp sang `train_ext` (900 da). Bộ này không dùng cho kết quả nào.

**Các bước GPU tùy chọn.** Không cần cho việc huấn luyện mạng.

| Bước | Lệnh (trong `exp2_layered/`) | Thời gian |
|---|---|---|
| 01 kiểm bộ mô phỏng | `01_simulator_checks/code/sim_capabilities.py`, `sim_capabilities_2.py` | vài phút |
| 02 mô hình thuận | `02_forward_model/code/nuisance_runs.py`, `layered_runs.py` (rồi `*_analysis.py` tương ứng trên CPU) | vài giờ |
| 05 giới hạn thông tin 14 tham số | `05_error_analysis/code/crlb_runs.py` (rồi `crlb_analysis.py` trên CPU) | khoảng 18 giờ |

## 4. Từ dữ liệu ra mô hình (CPU)

Trong `exp2_layered/`:

```powershell
$PY = "..\.venv-analysis\Scripts\python.exe"
foreach ($s in 42, 43, 44) { & $PY common\cnn_variants.py hetero $s }    # mô hình cuối, ~13 phút mỗi seed
foreach ($s in 42, 43, 44) { & $PY common\cnn_variants.py wide $s }      # mô hình về nhì
& $PY 04_training\code\compare_validation_ext.py                          # chấm mọi mô hình trên validation_ext
```

Mô hình được lưu vào `04_training/results/variants/`. Các biến thể khác huấn luyện tương tự: `base`, `sched`, `multi`, thêm `--nskin=N` cho đường cong học, `--hisnr` (cần `train_hisnr`).

Bộ test khóa đã được chấm một lần bằng các mô hình đang lưu (`06_final_test/results/locked_test/OPENED.json`); `score_locked_test.py` sẽ từ chối chấm lại.

## 5. Kiểm dữ liệu tạo lại

```powershell
.\.venv-analysis\Scripts\python tools\compare_regenerated.py <đường dẫn repo gốc> .
```

Công cụ so từng file MC có ở cả hai nơi:
- cấu hình và seed của lần chạy, hoặc cấu hình làn da: phải trùng;
- phản xạ trung bình hoặc N0: lệch dưới 0,5 %;
- profile (gộp 1 mm trên ±16 mm): độ lệch phải dưới 3 lần mức nhiễu MC dự kiến giữa hai lần chạy độc lập;
- manifest: phải trùng;
- bảng tra và bộ F/K: lệch dưới 2 %.

Cùng loại GPU thì file gần như trùng khít. Khác GPU thì chuỗi photon khác nhau, nên file chỉ khớp trong phạm vi nhiễu MC.

Ví dụ một lần chạy thử: 2 lần chạy của thử nghiệm 1 tạo lại trên RTX 3060 (bản gốc làm trên AMD RX 6600), và 1 làn da của thử nghiệm 2:

```
OK   exp1_homogeneous/01_simulator_checks/results/N0.npz: same config/seed True, mean diff 9.28e-06, profile RMS diff 1.03e-03 (limit 3.38e-03)
OK   exp1_homogeneous/01_simulator_checks/results/channel_2mm.npz: same config/seed True, mean diff 6.57e-05, profile RMS diff 1.51e-03 (limit 4.89e-03)
OK   exp2_layered/03_dataset/data/train/train_000.npz: same config True, N0 diff 1.32e-05, worst vein RMS diff 1.33e-04 (limit 1.10e-02)
```

Mã thoát bằng 0 chỉ khi mọi file được so đều đạt.
