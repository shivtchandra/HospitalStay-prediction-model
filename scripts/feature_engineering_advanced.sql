-- This query is a conceptual example of building a SAPS-II score.
-- NOTE: This is a complex clinical score and this query is a simplified representation.
-- A full implementation requires careful handling of units, time windows, and edge cases.

WITH first_day_vitals AS (
    -- Get the worst vital signs in the first 24 hours
    SELECT
        p.subject_id,
        p.hadm_id,
        MIN(CASE WHEN ce.itemid = 220045 THEN ce.valuenum ELSE NULL END) as min_hr,
        MAX(CASE WHEN ce.itemid = 220045 THEN ce.valuenum ELSE NULL END) as max_hr,
        MIN(CASE WHEN ce.itemid = 220210 THEN ce.valuenum ELSE NULL END) as min_sbp,
        MAX(CASE WHEN ce.itemid = 220210 THEN ce.valuenum ELSE NULL END) as max_sbp,
        MAX(CASE WHEN ce.itemid = 220277 THEN ce.valuenum ELSE NULL END) as max_temp_c,
        -- ... and so on for all other required vitals
    FROM admissions p
    LEFT JOIN chartevents ce ON p.hadm_id = ce.hadm_id
    WHERE ce.charttime BETWEEN p.admit_dt AND p.admit_dt + INTERVAL '1 day'
    GROUP BY p.subject_id, p.hadm_id
),
first_day_labs AS (
    -- Get the worst lab values in the first 24 hours
    SELECT
        p.hadm_id,
        MAX(CASE WHEN le.itemid = 50882 THEN le.valuenum ELSE NULL END) as max_bicarbonate,
        MAX(CASE WHEN le.itemid = 50821 THEN le.valuenum ELSE NULL END) as max_potassium,
        -- ... and so on for PaO2, FiO2, Bilirubin, etc.
    FROM admissions p
    LEFT JOIN labevents le ON p.hadm_id = le.hadm_id
    WHERE le.charttime BETWEEN p.admit_dt AND p.admit_dt + INTERVAL '1 day'
    GROUP BY p.hadm_id
)
-- The full query would then use complex CASE WHEN statements to assign points
-- for each variable according to the official SAPS-II methodology and sum them up.
SELECT
    adm.subject_id,
    adm.hadm_id,
    -- ... other features ...
    -- NEW FEATURE: A clinically validated risk score
    calculate_saps_ii_score(vitals.*, labs.*, pat.anchor_age) AS saps_ii_score
FROM
    admissions adm
-- ... joins ...
