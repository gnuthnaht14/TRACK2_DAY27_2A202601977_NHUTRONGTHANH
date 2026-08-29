# Incident Report — Data Reliability Game Day

## Severity
**P1 (Critical Business Impact)** — CEO Dashboard hiển thị số liệu doanh thu sai lệch và Customer Support Bot trích xuất chính sách hoàn tiền lỗi thời.

## Summary
Vào ngày 29/08/2026, hệ thống e-commerce ghi nhận pipeline chạy `SUCCESS` nhưng dữ liệu downstream bị suy giảm chất lượng nghiêm trọng:
1. Doanh thu hàng ngày bị tính toán sai lệch do duplicate order primary keys và customer dimension SCD Type 2 join inflation.
2. Lưu lượng đơn hàng incoming bị sụt giảm đột ngột (partial ingestion drop 75%) không kích hoạt SQL error truyền thống.
3. Tài liệu Knowledge Base của hệ thống Support RAG bị trễ hạn cập nhật (stale timestamps > 3h), dẫn đến việc bot hỗ trợ khách hàng trả lời chính sách refund cũ.

## Detection
- **Signal 1 (Contract Breach)**: `orders_contract.yaml` vi phạm check `unique` trên cột `order_id` (severity: critical) và vi phạm `freshness` trên `updated_at` / `published_at` (severity: warning).
- **Signal 2 (Anomaly Detection)**: Anomaly detector phát hiện `row_count` giảm mạnh xuống 25% (Z-score & MAD modified score vượt ngưỡng 3.5).
- **Signal 3 (SLO Burn Rate)**: Burn rate của `critical_contract_pass` vượt 15.0x trên cả short và long window, kích hoạt cảnh báo Page khẩn cấp cho On-call Engineer.
- **Signal 4 (dbt Tests)**: Singular test `assert_revenue_matches_stg` và dbt unit tests phát hiện revenue inflation khi join với duplicate active customer records.

## Root Cause
1. **Upstream Ingestion Duplicate & Truncation**: Hệ thống upstream CDC gửi trùng lặp batch `order_id` và bị ngắt kết nối giữa chừng trong quá trình sync đơn hàng.
2. **Dimension Modeling Gap**: Bảng `stg_customers` chứa nhiều hơn 1 bản ghi `is_active = true` cho cùng một `customer_id` (lỗi quản lý SCD Type 2), dẫn đến phép `LEFT JOIN` trong `fct_daily_revenue` nhân bản số lượng dòng đơn hàng và làm sai lệch doanh thu.
3. **Stale KB Sync Worker**: Tiến trình embedding/sync KB documents bị treo, khiến `published_at` bị trễ hơn 3 giờ so với ngưỡng SLA 60 phút.

## Evidence
1. **Contract Validation Output**:
   - `check="unique"`, `column="order_id"`, `severity="critical"`, `passed=False`, `details="duplicate_rows=6"`.
   - `check="freshness"`, `column="updated_at"`, `severity="warning"`, `passed=False`, `details="delay_minutes=180.0 > max_delay=30.0"`.
2. **Anomaly Engine Score**:
   - `metric_name="row_count"`, `method="auto"`, `is_anomaly=True`, `score=4.82` (MAD score), `reason="z_score=4.82, mad_score=5.12 [metric=row_count, dow=5]"`.
3. **dbt Unit Test Failure Evidence**:
   - Unit test `test_duplicate_active_customer_inflates_revenue_warning` chứng minh 1 đơn hàng $100 khi join với 2 bản ghi active customer sẽ bị nhân đôi thành $200.
4. **SLO Evaluation**:
   - Target = 99.9%, Actual Bad Rate = 100% (trong batch lỗi), Burn Rate = 1000x -> Kích hoạt Paging theo Multi-window policy.

## Blast Radius

```text
raw_orders / raw_customers / kb_documents (Upstream Ingestion)
  │
  ├──> stg_orders ──> fct_daily_revenue ──> CEO Dashboard (KPI Doanh thu bị sai lệch)
  │                                     └──> Finance & Accounting Reconciliation
  │
  ├──> stg_customers ──> Customer Analytics & CRM Segmentations
  │
  └──> active KB Documents ──> Vector Index ──> RAG Support Agent (Khách hàng nhận thông tin refund sai)
```

## Mitigation
1. **Immediate Quarantine & Block**:
   - Kích hoạt cơ chế `determine_action(issues) == 'block'` tại Data Contract Validator để chặn các batch đơn hàng chứa duplicate PK hoặc sai kiểu dữ liệu đi vào staging layer.
2. **Deduplication in Transformation**:
   - Áp dụng `ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY updated_at DESC)` trong staging để đảm bảo mỗi customer chỉ có duy nhất 1 bản ghi active tại một thời điểm.
3. **KB Sync Pipeline Restart**:
   - Trigger worker re-sync toàn bộ tài liệu KB mới nhất từ authoritative source.

## Recovery
- Chạy script `python scripts/reset_lab.py` để khôi phục data clean, đồng bộ dbt seeds và re-anchor timestamps.
- Chạy lại pipeline: `make baseline` và `make dbt`.

## Verification
- [x] Contract healthy: Toàn bộ 8 checks trên orders và KB contract đều đạt `passed=True`.
- [x] dbt tests healthy: 100% generic data tests, singular business tests và dbt native unit tests đều PASS.
- [x] Anomaly returned to expected range: `is_anomaly=False` cho cả row_count, distribution KS-test, và RAG text length.
- [x] SLO healthy: Burn rate <= 1.0, Error budget còn lại 100%.
- [x] Downstream output verified: CEO Dashboard hiển thị chính xác daily revenue khớp 100% với completed orders.

## Prevention / Action Items
| Action | Owner | Deadline | Why |
|---|---|---|---|
| Triển khai Pre-ingestion Data Contract Gate chặn duplicate PK và type drift | Data Platform Team | 2026-09-05 | Ngăn dữ liệu lỗi lọt vào data warehouse |
| Thêm dbt unit test bắt buộc cho tất cả các mô hình join dimension có SCD Type 2 | Analytics Engineers | 2026-09-07 | Chống lỗi phình doanh thu do join 1-nhiều |
| Cấu hình Multi-window Burn Rate Alerting qua PagerDuty / Slack Webhook | SRE / Observability | 2026-09-10 | Giảm thiểu false alarm và phản ứng nhanh với sự cố P1 |
| Thiết lập automated Freshness & Embedding Drift check trên Vector Database | AI Engineer | 2026-09-12 | Đảm bảo tri thức của Support Agent luôn cập nhật |
