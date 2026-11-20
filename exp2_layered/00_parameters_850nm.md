# Bảng tham số quang học có nguồn ở 850 nm

Các giá trị dưới đây được đọc từ nguồn; giá trị nào là phép tính của dự án được ghi rõ. Dòng "còn thiếu" phải bổ sung trước khi dựng mô hình da nhiều lớp (bước 01).

## Máu toàn phần (Hct 45 %)

Nguồn: Bosschaart N, Edelman GJ, Aalders MCG, van Leeuwen TG, Faber DJ. *A literature review and novel theoretical approach on the optical properties of whole blood.* Lasers Med Sci 29:453–479 (2014). doi:10.1007/s10103-013-1446-7 (PMC3953607), bảng phụ lục "Tabulated data for the optical property spectra of whole blood, hct = 45 %".

| Đại lượng ở 850 nm | Giá trị | Ghi chú |
| --- | --- | --- |
| μa, SO₂ > 98 % | 0,49 mm⁻¹ | Phổ trung bình từ tài liệu |
| μa, SO₂ = 0 % | 0,42 mm⁻¹ | |
| μs, SO₂ > 98 % (tổng hợp tài liệu) | 74,20 mm⁻¹ | |
| g, SO₂ > 98 % (tổng hợp tài liệu) | 0,9832 | |
| μs (tính theo lý thuyết KK/PY), SO₂ > 98 % / 0 % | 64,33 / 56,30 mm⁻¹ | |
| g (lý thuyết), SO₂ > 98 % / 0 % | 0,9798 / 0,9806 | |
| μs′ = μs(1 − g) | ≈ 1,25 mm⁻¹ (tổng hợp); ≈ 1,30 mm⁻¹ (lý thuyết) | **Dự án tính** từ công thức của bài |
| Đổi hematocrit cho μs | Phương trình 13 và 17 của bài; dạng gần đúng γ(hct) ≈ (1 − hct)². Bản HTML hiển thị số mũ không rõ; cần đối chiếu bản PDF trước khi dùng. | |

So với dự án hiện tại: μa kênh chuẩn 0,508 mm⁻¹ (tính từ phổ Prahl, Hb 150 g/L, SO₂ 0,70) nằm trong cùng cỡ với 0,42–0,49 mm⁻¹ của Bosschaart. Kênh hiện tại **không tán xạ**; máu thật có μs′ ≈ 1,25 mm⁻¹ với g ≈ 0,98.

## Da

Nguồn: Jacques SL. *Skin Optics.* Oregon Medical Laser Center News, 1/1998 (omlc.org/news/jan98/skinoptics.html). Tác giả tự ghi đây là ước lượng gần đúng. Bài tổng quan đầy đủ hơn — Jacques SL, *Optical properties of biological tissues: a review*, Phys Med Biol 58:R37 (2013), doi:10.1088/0031-9155/58/11/R37 — **chưa đọc được toàn văn** (tường phí); cần bản tác giả hoặc truy cập thư viện.

| Đại lượng | Công thức của nguồn | Giá trị ở 850 nm (dự án tính) |
| --- | --- | --- |
| μa melanosome | 6,6 × 10¹¹ · λ^−3,33 cm⁻¹ (λ tính bằng nm) | 116 cm⁻¹ = 11,6 mm⁻¹ |
| Tỉ lệ thể tích melanosome trong biểu bì | Da sáng 1,3–6,3 %; trung bình 11–16 %; sẫm 18–43 % (giả định biểu bì 60 µm) | — |
| μa biểu bì | f_mel · μa_mel + (1 − f_mel) · μa_nền | 0,18–5,0 mm⁻¹ tùy màu da |
| μa nền da không máu | 0,244 + 85,3 · exp(−(λ − 154)/66,2) cm⁻¹ | 0,025 mm⁻¹ |
| μs′ da (Rayleigh + Mie sợi collagen) | 2 × 10¹² λ⁻⁴ + 2 × 10⁵ λ^−1,5 cm⁻¹ | 1,19 mm⁻¹ |
| g da | 0,7–0,95, phụ thuộc bước sóng | — |
| Độ dày biểu bì | 60 µm (giá trị giả định của nguồn) | — |
| Đám rối tĩnh mạch nông | Cách bề mặt khoảng 100–200 µm; f_blood trung bình khoảng 0,2 % | — |

So với dự án hiện tại: nền μs′ = 1,0 mm⁻¹ gần với 1,19 mm⁻¹ ở trên; nền μa = 0,010 mm⁻¹ thấp hơn 0,025 mm⁻¹ của da không máu.

## Độ sâu và kích thước tĩnh mạch nông (siêu âm, người)

Theo các bài trên PubMed:

| Nguồn | Nội dung liên quan |
| --- | --- |
| Mukai K và cs., J Vasc Access 21:900 (2020), doi:10.1177/1129729820909187 | Tĩnh mạch nông chi trên; độ sâu, màu da (b*) và giới ảnh hưởng mạnh đến khả năng nhìn thấy; ngưỡng độ sâu 2,3 mm (AUC 0,91). |
| Ichimura M và cs., Clin Anat 33:440 (2020), doi:10.1002/ca.23559 | Tĩnh mạch khó sờ: độ sâu 4,57 ± 2,31 mm trước vỗ, 4,23 ± 2,41 mm sau vỗ. |
| Putensen D và cs., J Clin Apher 31:501 (2016), doi:10.1002/jca.21440 | Tĩnh mạch được đặt kim dưới siêu âm: độ sâu 2–12 mm, trung bình 5 mm. |
| Sharp R và cs., Appl Nurs Res 42:83 (2018), doi:10.1016/j.apnr.2018.06.013 | Đo đường kính và độ sâu tĩnh mạch đầu và tĩnh mạch giữa khuỷu bằng siêu âm. |

Hàm ý: dải 0,5–5 mm của dự án phủ nhóm tĩnh mạch nhìn thấy và một phần nhóm khó; tĩnh mạch rất sâu (> 5 mm) nằm ngoài dải.

## Công trình gần nhất về độ sâu tĩnh mạch bằng NIR

| Nguồn | Nội dung |
| --- | --- |
| Bachir W, Abo Dargham F. Phys Eng Sci Med 45:135 (2022), doi:10.1007/s13246-021-01096-x | Laser 830 nm, phantom da sẫm có ống chứa máu; hồi quy tuyến tính mức xám theo độ sâu tới khoảng 3 mm (R = 0,88). Đây là đối chứng "dựa trên đặc trưng" cần so sánh. |
| Mela CA và cs., Int J CARS 14:203 (2019), doi:10.1007/s11548-018-1865-9 | Hệ chụp tĩnh mạch kết hợp NIR truyền qua và ảnh khả kiến; phát hiện ống silicone dưới 3 mm mô. |
| Kimori K và cs., SAGE Open Med 3 (2015), doi:10.1177/2050312115615365 | Thiết bị soi tĩnh mạch ở người cao tuổi; đo độ sâu và đường kính tĩnh mạch không nhìn thấy. |

## Ảnh phân cực

| Nguồn | Nội dung |
| --- | --- |
| Jacques SL, Ramella-Roman JC, Lee K. J Biomed Opt 7:329 (2002), doi:10.1117/1.1484498 | Ảnh phân cực song song/vuông góc trên da; ảnh tỉ số phân cực chọn photon nông, ảnh vuông góc chủ yếu là photon đi sâu vào bì. |
| Ramella-Roman JC và cs., J Biomed Opt 9:1305 (2004), doi:10.1117/1.1781667 | Camera phân cực cầm tay chi phí thấp (tấm phân cực tấm mỏng, LED). |

## Mô hình kiểm chứng

| Nguồn | Dùng cho |
| --- | --- |
| Farrell TJ, Patterson MS, Wilson B. Med Phys 19:879 (1992), doi:10.1118/1.596777 | Lời giải khuếch tán cho phản xạ theo bán kính; kiểm mô hình thuận bước 01. |

## Bổ sung cho bước 02 — mô hình da nhiều lớp

Nguồn chính là Jacques SL, *Optical properties of biological tissues: a review*, Phys Med Biol 58:R37 (2013), doi:10.1088/0031-9155/58/11/R37. Bản tác giả lấy tại omlc.org/news/dec14/Jacques_PMB2013. Bảng 1 dùng công thức μs′(λ) = a·(λ/500 nm)^−b. Cột μs′ ở 850 nm là **dự án tính** từ a và b, với (850/500) = 1,7.

| Dòng trong bảng | a (cm⁻¹) | b | μs′ ở 850 nm (mm⁻¹) |
| --- | --- | --- | --- |
| Epidermis (Salomatina 2006) | 68,7 | 1,161 | 3,71 |
| Dermis (Salomatina 2006) | 45,3 | 1,292 | 2,28 |
| Skin (Simpson 1998) | 37,2 | 1,390 | 1,78 |
| Skin, trung bình | 46,0 | 1,421 | 2,16 |
| Subcutaneous fat (Simpson 1998) | 13,7 | 0,385 | 1,12 |
| Subcut. fat (Salomatina 2006) | 35,2 | 0,988 | 2,08 |
| Fatty tissue, trung bình | 18,4 | 0,672 | 1,29 |
| Muscle (Tromberg 1996) | 13,0 | 0,926 | 0,80 |

Bảng 3 của Jacques 2013 (tham số hấp thụ in vivo):

- Bì (Choudhury 2010): thể tích máu B = 0,20 %, S = 39 %, nước W = 65 %.
- Biểu bì: tỉ lệ melanosome M = 2,5 %.
- Da type I–II, 600–1000 nm (Tseng 2011): M = 0,87 %. Da type V–VI: M = 1,65 %.
- Cẳng tay (Matcher 1997): B = 5,03 %, S = 64 %. Số này là trung bình của cả khối mô sâu.

Mỡ (Simpson 1998, doi:10.1088/0031-9155/43/9/003): ở 633 nm, μa = 0,013 mm⁻¹ và μs′ = 1,26 mm⁻¹. Bì có tính chất quang khoảng gấp đôi mỡ.

Độ dày da: siêu âm tần số cao ở mặt trong đoạn gần cẳng tay cho da trung bình 1,19 mm, dải 0,65–1,55 mm (Van Mulder TJS và cs., Vaccine 35:1810, 2017; doi:10.1016/j.vaccine.2016.07.039).

Giá trị dùng trong bước 02 (`exp2_layered/02_forward_model/code/layered_runs.py`):

| Lớp | Độ dày | μa (mm⁻¹) | μs′ (mm⁻¹) |
| --- | --- | --- | --- |
| Biểu bì | 0,06 mm | 0,37 (M = 3 %); da sẫm M = 15 % → 1,76 | 3,71 |
| Bì | 1,1 mm | 0,02 (**ước lượng**) | 2,28 |
| Mỡ | 6 mm (**giả định**) | 0,010 (**ước lượng**) | 1,29 |
| Cơ | phần còn lại | 0,026 (**ước lượng**) | 0,80 |
| Máu tĩnh mạch | Ø 3 mm | 0,46 | μs = 74,2; g = 0,983 |

Các lớp mô dùng g = 0,9 và n = 1,37.

## Còn thiếu

- Độ dày mỡ dưới da ở cẳng tay: Sieg 2003 (Int J Oral Maxillofac Surg 32:544) có đo, nhưng chưa đọc được bảng số.
- μa của mỡ và của bì ở đúng 850 nm (phổ trong Simpson 1998 và Salomatina 2006).
- Kiểm lại số mũ của công thức đổi hematocrit (Bosschaart, phương trình 17) trên bản PDF.
- Các công trình tái hiện được (bảng tham số.2): ứng viên đầu là Bachir & Abo Dargham 2022.
