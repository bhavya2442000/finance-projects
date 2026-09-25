-- Cumulative net charge-off curves by vintage (issue quarter) and grade.
-- The file has no charge-off date, so it is estimated as last payment month + 5:
-- LendingClub charged off loans at ~150 days past due (assumption, see README).
CREATE OR REPLACE TABLE vintage_curves AS
WITH bad AS (
    SELECT
        date_trunc('quarter', issue_date)                               AS vintage_q,
        grade,
        LEAST(36, GREATEST(1, date_diff('month', issue_date,
              COALESCE(out_last_pymnt_date, issue_date)) + 5))          AS chargeoff_mob,
        out_ead - (out_recoveries - out_recovery_fee)                   AS net_loss
    FROM loans
    WHERE is_bad = 1
),
funded AS (
    SELECT date_trunc('quarter', issue_date) AS vintage_q, grade,
           SUM(funded_amnt) AS funded, COUNT(*) AS n_loans
    FROM loans GROUP BY ALL
),
grid AS (
    SELECT f.vintage_q, f.grade, f.funded, f.n_loans, m.mob
    FROM funded f CROSS JOIN range(1, 37) AS m(mob)
),
loss_by_mob AS (
    SELECT vintage_q, grade, chargeoff_mob AS mob, SUM(net_loss) AS net_loss
    FROM bad GROUP BY ALL
)
SELECT
    g.vintage_q, g.grade, g.mob, g.n_loans, g.funded,
    SUM(COALESCE(l.net_loss, 0)) OVER (
        PARTITION BY g.vintage_q, g.grade ORDER BY g.mob)               AS cum_net_loss,
    SUM(COALESCE(l.net_loss, 0)) OVER (
        PARTITION BY g.vintage_q, g.grade ORDER BY g.mob) / g.funded    AS cum_net_loss_rate
FROM grid g
LEFT JOIN loss_by_mob l USING (vintage_q, grade, mob)
ORDER BY 1, 2, 3;

-- Loss given default by grade: share of exposure not recovered after charge-off.
CREATE OR REPLACE TABLE lgd_by_grade AS
SELECT
    grade,
    COUNT(*)                                                            AS n_defaults,
    SUM(out_ead)                                                        AS ead,
    SUM(out_recoveries - out_recovery_fee)                              AS net_recoveries,
    1 - SUM(out_recoveries - out_recovery_fee) / SUM(out_ead)           AS lgd
FROM loans
WHERE is_bad = 1 AND vintage_year <= 2014
GROUP BY grade
ORDER BY grade;
