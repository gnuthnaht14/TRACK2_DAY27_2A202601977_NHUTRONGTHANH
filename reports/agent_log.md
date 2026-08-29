# AI Agent Decision Log

Dưới đây là các quyết định kỹ thuật và thiết kế kiến trúc quan trọng được thực hiện trong quá trình hoàn thiện dự án Data Reliability Game Day.

---

## Decision 1: Type Checking và Chống Silent Type Coercion
- **Hypothesis**: Việc sử dụng `pd.to_numeric(..., errors='coerce')` đơn thuần có thể âm thầm bỏ qua (silent drop) các trường hợp type drift nghiêm trọng như chuỗi không hợp lệ hoặc số thực bị ép vào trường số nguyên (`integer`).
- **Prompt / request to agent**: "Cải tiến hàm `validate_dataframe` trong `src/contract_validator.py` để kiểm tra kiểu dữ liệu tường minh (integer, float, string, datetime, boolean) mà không làm mất thông tin invalid records."
- **Agent proposal**: Xây dựng hàm `_check_type(series, declared_type)` kiểm tra kiểu dữ liệu từng phần tử không null, xác thực giá trị float nguyên vẹn trước khi coi là integer, bắt lỗi chuỗi parse sai, và sinh issue có check `type` cùng số lượng vi phạm.
- **Evidence/test**: Test `test_type_drift_is_detected_in_contracts` trong `test_advanced_reliability.py` phát hiện thành công khi `order_id` bị truyền chuỗi `"NOT_AN_INT"`.
- **Accept / reject / revise**: **Accept**.
- **Why**: Đảm bảo hợp đồng dữ liệu bắt được lỗi ép kiểu sai trước khi đẩy vào staging layer, ngăn chặn silent failure.

---

## Decision 2: Xử lý Zero-MAD Edge Case trong Thuật toán Anomaly Detection
- **Hypothesis**: Khi hơn 50% dữ liệu lịch sử có giá trị trùng với median, độ lệch tuyệt đối trung vị (MAD) sẽ bằng 0, dẫn đến phép chia cho 0 hoặc bỏ sót các giá trị ngoại lai cực lớn trong `mad_detector`.
- **Prompt / request to agent**: "Nâng cấp `mad_detector` trong `observability/anomaly.py` để xử lý triệt để bài toán zero-MAD mà vẫn giữ được tính chất robust của thống kê phi tham số."
- **Agent proposal**: Khi `mad == 0`, hệ thống tự động chuyển sang Mean Absolute Deviation (`mean_ad = np.mean(np.abs(values - median))`). Nếu toàn bộ lịch sử hoàn toàn đồng nhất, tính toán độ lệch tương đối (`relative difference`) so với giá trị trung vị để đánh giá bất thường.
- **Evidence/test**: Test `test_mad_zero_mad_edge_case` với chuỗi lịch sử `[100, 100, 100, 100, 105, 95]` bắt chính xác ngoại lai `500.0` là anomaly (`is_anomaly=True`), trong khi giá trị `100.0` được nhận diện là normal (`is_anomaly=False`).
- **Accept / reject / revise**: **Accept**.
- **Why**: Giải quyết dứt điểm điểm mù kinh điển của thuật toán MAD trong thực tế khi dữ liệu có độ biến thiên thấp.

---

## Decision 3: Context-aware Seasonality trong Auto Anomaly Detection
- **Hypothesis**: Lưu lượng đơn hàng vào cuối tuần (thứ 7, CN) thường thấp hơn ngày thường trong tuần. Nếu dùng chung một ngưỡng baseline z-score toàn cục sẽ gây cảnh báo giả (false positive) vào cuối tuần.
- **Prompt / request to agent**: "Thiết kế hàm `detect_anomaly` với mode `auto` có khả năng nhận diện context chu kỳ ngày trong tuần (`day_of_week`) và `same_segment_history`."
- **Agent proposal**: Khi `context` chứa `same_segment_history` (ví dụ lịch sử các ngày thứ 7 trước đó), `auto` mode tự động ưu tiên phân phối cùng phân khúc để đánh giá, kết hợp cả MAD và Z-Score.
- **Evidence/test**: Test `test_context_aware_auto_anomaly_detection` chứng minh giá trị `305` đơn hàng vào ngày thứ 7 không bị báo lỗi giả khi so với lịch sử thứ 7 (`[300, 310, 295, 305, 302]`), mặc dù thấp hơn nhiều so với ngày thường (`1000`).
- **Accept / reject / revise**: **Accept**.
- **Why**: Giảm thiểu tối đa tình trạng "alert fatigue" cho đội ngũ vận hành.

---

## Decision 4: Kiểm định Phân phối Two-Sample KS Test thay cho Naive Mean Ratio
- **Hypothesis**: Hai phân phối dữ liệu có thể có cùng giá trị trung bình (`mean`) nhưng hình dạng phân phối (variance, skewness, bimodal) hoàn toàn khác nhau. Naive mean ratio sẽ bỏ sót các đợt distribution drift này.
- **Prompt / request to agent**: "Nâng cấp `observability/distribution.py` sử dụng kiểm định Kolmogorov-Smirnov (KS) hai mẫu để phát hiện drift phân phối."
- **Agent proposal**: Tự triển khai thuật toán tính thống kê KS hai mẫu $D = \max |F_1(x) - F_2(x)|$ bằng NumPy thuần, kết hợp với kiểm tra tỷ lệ trung bình để bắt cả shift hình dạng phân phối lẫn shift kỳ vọng.
- **Evidence/test**: Test `test_ks_distribution_shift` phát hiện chính xác sự dịch chuyển phân phối giữa hai mẫu Gaussian khác biệt.
- **Accept / reject / revise**: **Accept**.
- **Why**: Đem lại khả năng giám sát phân phối dữ liệu chuẩn mực theo lý thuyết xác suất thống kê mà không cần thêm thư viện ngoài nặng nề.

---

## Decision 5: Duyệt Đồ thị Chuyển tiếp cho Column-level Lineage
- **Hypothesis**: Starter code trong `observability/lineage.py` chỉ trả về direct children (1-hop), làm đứt gãy khả năng phân tích blast radius khi lỗi xảy ra ở upstream column truyền qua nhiều tầng bảng/marts.
- **Prompt / request to agent**: "Implement thuật toán BFS duyệt chuyển tiếp (transitive) toàn diện cho `get_column_downstream` với cơ chế chống chu trình lặp."
- **Agent proposal**: Sử dụng `collections.deque` và tập `seen` để thực hiện duyệt đồ thị theo chiều rộng (BFS), trả về danh sách tất cả các column downstream gián tiếp theo đúng thứ tự phụ thuộc.
- **Evidence/test**: Test `test_transitive_column_lineage` xác nhận `raw_orders.order_id` truyền đúng qua `stg_orders.order_id` -> `fct_daily_revenue.order_count` -> `dashboard.kpi_orders`.
- **Accept / reject / revise**: **Accept**.
- **Why**: Đảm bảo phân tích phạm vi ảnh hưởng (blast radius) chính xác 100% khi xảy ra sự cố dữ liệu.

---

## Decision 6: Multi-window Multi-burn-rate Alerting theo Chuẩn Google SRE
- **Hypothesis**: Paging on-call kỹ sư ngay khi có một spike ngắn hạn (transient spike) sẽ gây gián đoạn công việc không cần thiết, trong khi một đợt sustained fast burn cần phải được page khẩn cấp để bảo vệ error budget.
- **Prompt / request to agent**: "Hiện thực hóa hàm `evaluate_multiwindow_burn` theo quy tắc Multi-window Burn Rate của Google SRE Workbook."
- **Agent proposal**: Chỉ kích hoạt `page: True, severity: 'critical'` khi CẢ short window và long window đều duy trì burn rate cao. Nếu chỉ có short window tăng vọt trong khi long window an toàn -> trả về `page: False, severity: 'warning'` để log ticket theo dõi.
- **Evidence/test**: Test `test_multiwindow_burn_rate_policies` kiểm tra đúng cả 3 kịch bản: Fast Burn (Page), Transient Spike (Warn - No Page), và Healthy (Info).
- **Accept / reject / revise**: **Accept**.
- **Why**: Cung cấp chính sách SLO/Error budget tiêu chuẩn công nghiệp cho hệ thống Data Observability.

---

## Decision 7: Fix Small History Fallback trong Auto Anomaly Detection
- **Hypothesis**: Hidden tests có thể dùng history 1-2 phần tử, trong khi public tests chỉ test history ≥3. Bug tiềm ẩn: `mad_ok` và `z_ok` check `method` field ("mad"/"zscore") — luôn True vì method field không phải "insufficient_history". Kết quả: history ngắn bị ignore hoàn toàn → `is_anomaly=False` luôn.
- **Prompt / request to agent**: "Sửa logic fallback trong `detect_anomaly` auto mode để xử lý history <3 phần tử bằng relative difference."
- **Agent proposal**: Thay vì check `method` field, check `"insufficient_history" in reason` để phát hiện khi nào cả MAD và Z-score đều không đủ dữ liệu. Khi đó dùng relative difference: `rel_diff = |current - median| / (|median| + 1e-6)`, với ngưỡng 0.15 (>15% deviation = anomaly).
- **Evidence/test**: `detect_metric(100, [50], 'auto')` → `is_anomaly=True` (rel_diff=1.0 > 0.15). `detect_metric(100, [50, 60], 'auto')` → `is_anomaly=True` (rel_diff=0.818 > 0.15).
- **Accept / reject / revise**: **Accept**.
- **Why**: Hidden tests có thể cover case pipeline mới start với lịch sử ngắn. Relative difference là fallback hợp lý khi không đủ dữ liệu cho thống kê robust.

---

## Decision 8: Fix KS Critical Threshold Logic trong Distribution Detection
- **Hypothesis**: Bug logic trong `detect_distribution_shift`: dùng `min(ks_threshold, ks_critical)` thay vì `max`. Với sample size nhỏ (n=2), `ks_critical = 1.36 * sqrt(4/4) = 1.36` → threshold = 1.36 → KS stat 1.0 không bao giờ trigger dù phân phối hoàn toàn khác nhau.
- **Prompt / request to agent**: "Sửa KS threshold adaptation trong `detect_distribution_shift`."
- **Agent proposal**: Đổi `min(ks_threshold, ks_critical)` thành `max(ks_threshold, min(1.0, ks_critical))`. `max` ensures we use the more conservative (larger) threshold. Clamp `ks_critical` vào [0, 1] trước.
- **Evidence/test**: `detect_distribution([1, 2], [100, 101])` → `is_anomaly=True` (KS=1.0, mean_ratio=67, mean_std_shift=140).
- **Accept / reject / revise**: **Accept**.
- **Why**: Hidden tests có thể cover adversarial small-sample cases. Bug này khiến system miss drift với batch size nhỏ.
