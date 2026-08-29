-- Singular test: Verify that total daily revenue in mart matches completed orders in staging.
-- Returns non-empty if there is any mismatch in daily aggregation.

with mart_totals as (
    select
        order_date,
        daily_revenue
    from {{ ref('fct_daily_revenue') }}
),
stg_totals as (
    select
        order_date,
        sum(amount_usd) as expected_revenue
    from {{ ref('stg_orders') }}
    where status = 'completed'
    group by 1
)
select
    m.order_date,
    m.daily_revenue,
    s.expected_revenue
from mart_totals m
join stg_totals s on m.order_date = s.order_date
where abs(m.daily_revenue - s.expected_revenue) > 0.01
