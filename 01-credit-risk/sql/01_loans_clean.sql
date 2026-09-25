-- One row per 36-month loan issued 2012-2015, with a final outcome.
-- Only fields known at origination go into features; outcome fields are
-- kept separately (prefixed out_) for loss and return analysis, never for modeling.
CREATE OR REPLACE TABLE loans AS
WITH base AS (
    SELECT
        id,
        strptime(issue_d, '%b-%Y')::DATE                                AS issue_date,
        year(strptime(issue_d, '%b-%Y'))                                AS vintage_year,
        -- application / bureau data at origination
        CAST(loan_amnt AS DOUBLE)                                       AS loan_amnt,
        CAST(funded_amnt AS DOUBLE)                                     AS funded_amnt,
        CAST(int_rate AS DOUBLE)                                        AS int_rate,
        CAST(installment AS DOUBLE)                                     AS installment,
        grade,
        sub_grade,
        CASE WHEN emp_length IS NULL OR emp_length = 'n/a' THEN NULL
             WHEN emp_length LIKE '<%' THEN 0
             ELSE CAST(regexp_extract(emp_length, '(\d+)', 1) AS INT) END AS emp_length_yrs,
        home_ownership,
        CAST(annual_inc AS DOUBLE)                                      AS annual_inc,
        verification_status,
        purpose,
        addr_state,
        CAST(dti AS DOUBLE)                                             AS dti,
        CAST(delinq_2yrs AS DOUBLE)                                     AS delinq_2yrs,
        date_diff('month', strptime(earliest_cr_line, '%b-%Y'),
                  strptime(issue_d, '%b-%Y'))                           AS credit_history_months,
        (CAST(fico_range_low AS DOUBLE) + CAST(fico_range_high AS DOUBLE)) / 2 AS fico,
        CAST(inq_last_6mths AS DOUBLE)                                  AS inq_last_6mths,
        CAST(mths_since_last_delinq AS DOUBLE)                          AS mths_since_last_delinq,
        CAST(open_acc AS DOUBLE)                                        AS open_acc,
        CAST(pub_rec AS DOUBLE)                                         AS pub_rec,
        CAST(revol_bal AS DOUBLE)                                       AS revol_bal,
        CAST(revol_util AS DOUBLE)                                      AS revol_util,
        CAST(total_acc AS DOUBLE)                                       AS total_acc,
        CAST(mort_acc AS DOUBLE)                                        AS mort_acc,
        CAST(pub_rec_bankruptcies AS DOUBLE)                            AS pub_rec_bankruptcies,
        CAST(tot_cur_bal AS DOUBLE)                                     AS tot_cur_bal,
        CAST(total_rev_hi_lim AS DOUBLE)                                AS total_rev_hi_lim,
        CAST(acc_open_past_24mths AS DOUBLE)                            AS acc_open_past_24mths,
        CAST(bc_util AS DOUBLE)                                         AS bc_util,
        CAST(num_actv_rev_tl AS DOUBLE)                                 AS num_actv_rev_tl,
        CAST(percent_bc_gt_75 AS DOUBLE)                                AS percent_bc_gt_75,
        CAST(mths_since_recent_inq AS DOUBLE)                           AS mths_since_recent_inq,
        CAST(pct_tl_nvr_dlq AS DOUBLE)                                  AS pct_tl_nvr_dlq,
        -- outcomes (not features)
        loan_status                                                     AS out_status,
        CAST(total_pymnt AS DOUBLE)                                     AS out_total_pymnt,
        CAST(total_rec_prncp AS DOUBLE)                                 AS out_rec_prncp,
        CAST(recoveries AS DOUBLE)                                      AS out_recoveries,
        CAST(collection_recovery_fee AS DOUBLE)                         AS out_recovery_fee,
        strptime(last_pymnt_d, '%b-%Y')::DATE                           AS out_last_pymnt_date
    FROM read_parquet('data/loans_raw.parquet')
    WHERE trim(term) = '36 months'
      AND issue_d IS NOT NULL
      AND year(strptime(issue_d, '%b-%Y')) BETWEEN 2012 AND 2015
      AND loan_status IN ('Fully Paid', 'Charged Off', 'Default', 'Late (31-120 days)')
)
SELECT
    *,
    CASE WHEN out_status = 'Fully Paid' THEN 0 ELSE 1 END              AS is_bad,
    installment * 12 / NULLIF(annual_inc, 0)                            AS pmt_to_income,
    loan_amnt / NULLIF(annual_inc, 0)                                   AS loan_to_income,
    -- principal still owed when the loan went bad = exposure at default
    CASE WHEN out_status <> 'Fully Paid'
         THEN funded_amnt - out_rec_prncp END                           AS out_ead,
    -- net cash back to the lender per dollar lent (before servicing fees and time value)
    (out_total_pymnt - funded_amnt) / funded_amnt                       AS out_net_return
FROM base;
