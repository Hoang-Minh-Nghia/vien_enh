# Kết luận kiểm chứng mô phỏng V2–V6

Phạm vi: Direct MC, ideal surface reflectance, one straight 3 mm channel, fixed background; no camera or phantom validation.

Điều kiện theo chất lượng MC: {'accepted': 60}.

Tệp này do chương trình sinh tự động theo quy tắc đã khóa; cần người đọc lại trước khi đưa vào luận văn.

## Chỉ số chính và các nhóm (tập điều kiện đạt chất lượng MC)

| Nhóm | Số điều kiện | MAE F (mm) | MAE K (mm) | K − F, TB ± SD (mm) | K − F theo seed 42/43/44 | Cả 3 seed F thấp hơn |
| --- | --- | --- | --- | --- | --- | --- |
| unseen_pooled | 20 | 0,159 | 0,358 | 0,199 ± 0,039 | 0,238 / 0,199 / 0,160 | Có |
| unseen_0.22 | 10 | 0,224 | 0,580 | 0,355 ± 0,056 | 0,409 / 0,358 / 0,298 | Có |
| unseen_0.40 | 10 | 0,094 | 0,136 | 0,042 ± 0,023 | 0,066 / 0,040 / 0,021 | Có |
| seen_weak_pooled | 20 | 0,257 | 0,628 | 0,371 ± 0,063 | 0,438 / 0,363 / 0,312 | Có |
| seen_weak_0.15 | 10 | 0,372 | 0,930 | 0,558 ± 0,087 | 0,654 / 0,538 / 0,483 | Có |
| seen_weak_0.30 | 10 | 0,143 | 0,327 | 0,184 ± 0,041 | 0,222 / 0,188 / 0,141 | Có |
| standard_0.508 | 10 | 0,075 | 0,037 | -0,038 ± 0,015 | -0,029 / -0,055 / -0,030 | Không |
| high_1.0 | 10 | 0,086 | 0,248 | 0,162 ± 0,044 | 0,164 / 0,117 / 0,205 | Có |
| all | 60 | 0,166 | 0,376 | 0,211 ± 0,033 | 0,248 / 0,198 / 0,186 | Có |

## Câu chữ kết luận theo mục 6 của kế hoạch

- F giảm MAE ở nhóm μa chưa học 0,22 / 0,40 ở cả ba seed. Kết luận về khái quát hóa chỉ dùng được nếu mục khả năng phân biệt độ sâu cũng đạt, và chỉ trong mô hình vật lý đã khảo sát.
- F giảm MAE trên MC mới ở nhóm 0,15 / 0,30 (mức F đã học); chưa đủ để nói khái quát sang μa mới.
- Đánh đổi tại mức chuẩn 0,508: MAE_F − MAE_K = 0,038 mm. F kém K ở mức này.

## Khả năng phân biệt độ sâu ở μa chưa học

| μa | Seed | Hệ số góc F | Chặn F (mm) | MAE F (mm) | Hệ số góc K | MAE K (mm) | MAE đoán hằng (mm) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.22 | 42 | 1,001 | 0,102 | 0,141 | 0,886 | 0,550 | 1,200 |
| 0.22 | 43 | 0,991 | 0,274 | 0,255 | 0,859 | 0,613 | 1,200 |
| 0.22 | 44 | 1,024 | 0,216 | 0,278 | 0,851 | 0,576 | 1,200 |
| 0.4 | 42 | 0,964 | 0,101 | 0,051 | 0,980 | 0,117 | 1,200 |
| 0.4 | 43 | 0,945 | 0,217 | 0,117 | 0,961 | 0,157 | 1,200 |
| 0.4 | 44 | 0,998 | 0,118 | 0,113 | 0,957 | 0,134 | 1,200 |

## Ổn định theo hấp thụ

Số cặp độ sâu–seed có R_F nhỏ hơn R_K và MAE_F cũng thấp hơn: 28/30. R nhỏ chỉ có ý nghĩa khi dự đoán cũng gần độ sâu thật.

## Mục tiêu MAE ≤ 0,5 mm (trung bình ba seed, không phải chuẩn lâm sàng)

F đạt 58/60 điều kiện; K đạt 43/60 điều kiện.

| Điều kiện chưa đạt ở ít nhất một nhánh | MAE F (mm) | MAE K (mm) |
| --- | --- | --- |
| z = 0.6, μa = 0.15 | 0,239 | 1,315 |
| z = 0.6, μa = 0.22 | 0,199 | 0,813 |
| z = 0.9, μa = 0.15 | 0,233 | 1,243 |
| z = 0.9, μa = 0.22 | 0,162 | 0,747 |
| z = 1.4, μa = 0.15 | 0,220 | 1,200 |
| z = 1.4, μa = 0.22 | 0,168 | 0,724 |
| z = 1.8, μa = 0.15 | 0,266 | 1,124 |
| z = 1.8, μa = 0.22 | 0,175 | 0,707 |
| z = 2.3, μa = 0.15 | 0,413 | 1,028 |
| z = 2.3, μa = 0.22 | 0,179 | 0,650 |
| z = 2.8, μa = 0.15 | 0,469 | 0,945 |
| z = 2.8, μa = 0.22 | 0,284 | 0,590 |
| z = 3.3, μa = 0.15 | 0,611 | 0,917 |
| z = 3.3, μa = 0.22 | 0,269 | 0,547 |
| z = 3.8, μa = 0.15 | 0,712 | 0,825 |
| z = 3.8, μa = 0.22 | 0,391 | 0,539 |
| z = 4.3, μa = 0.15 | 0,411 | 0,564 |

## Giới hạn

- Ba MC lặp đo biến thiên số học, ba seed CNN đo biến thiên huấn luyện; không nhân thành cỡ mẫu vật lý.
- Chỉ mô tả ghép cặp và độ nhất quán; không tuyên bố ý nghĩa thống kê quần thể.
- Chưa có camera hay phantom (V7–V8).
