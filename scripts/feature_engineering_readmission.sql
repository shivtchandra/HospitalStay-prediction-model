-- This query is designed to create a feature set for predicting 30-day readmission.

-- This CTE identifies the next admission date for each hospital stay for every patient.
WITH next_admission AS (
    SELECT
        subject_id,
        hadm_id,
        disch_dt,
        -- LEAD() is a window function that gets a value from a subsequent row.
        -- We partition by patient and order by their admission date to find the *next* admission.
        LEAD(admit_dt, 1) OVER (PARTITION BY subject_id ORDER BY admit_dt) AS next_admit_dt
    FROM
        admissions
)
-- We will join all our previous CTEs and features with this new information.
, lab_features AS (
    SELECT
        hadm_id,
        MAX(CASE WHEN itemid = 50912 THEN valuenum ELSE NULL END) AS max_creatinine,
        MIN(CASE WHEN itemid = 51222 THEN valuenum ELSE NULL END) AS min_hemoglobin
    FROM labevents GROUP BY hadm_id
),
vital_features AS (
    SELECT
        adm.hadm_id,
        AVG(CASE WHEN ce.itemid IN (220045, 211) THEN ce.valuenum ELSE NULL END) AS avg_heart_rate
    FROM admissions adm
    INNER JOIN chartevents ce ON adm.hadm_id = ce.hadm_id
    WHERE ce.charttime BETWEEN adm.admit_dt AND adm.admit_dt + INTERVAL '1 day'
    GROUP BY adm.hadm_id
)
-- Final SELECT statement
SELECT
    adm.subject_id,
    adm.hadm_id,
    
    -- THE NEW TARGET VARIABLE --
    -- Check if the next admission occurred within 30 days of the current discharge.
    CASE
        WHEN next_adm.next_admit_dt IS NOT NULL AND 
             (next_adm.next_admit_dt - adm.disch_dt) <= INTERVAL '30 days' THEN 1
        ELSE 0
    END AS was_readmitted_in_30_days,
    
    -- All our previously engineered features remain excellent predictors of risk
    pat.gender,
    pat.anchor_age,
    adm.admission_type,
    adm.insurance,
    did.long_title AS primary_diagnosis,
    COUNT(DISTINCT proc.icd_code) AS procedure_count,
    lab.max_creatinine,
    lab.min_hemoglobin,
    (pat.anchor_age * CASE WHEN vf.avg_heart_rate IS NULL THEN 0 ELSE ABS(vf.avg_heart_rate - 75) END) AS age_hr_interaction
    
FROM
    admissions adm
INNER JOIN
    patients pat ON adm.subject_id = pat.subject_id
LEFT JOIN
    diagnoses_icd diag ON adm.hadm_id = diag.hadm_id AND diag.seq_num = 1
LEFT JOIN
    d_icd_diagnoses did ON diag.icd_code = did.icd_code AND diag.icd_version = did.icd_version
LEFT JOIN
    procedures_icd proc ON adm.hadm_id = proc.hadm_id
LEFT JOIN
    lab_features lab ON adm.hadm_id = lab.hadm_id
LEFT JOIN
    vital_features vf ON adm.hadm_id = vf.hadm_id
LEFT JOIN
    next_admission next_adm ON adm.hadm_id = next_adm.hadm_id
    
GROUP BY
    adm.subject_id, adm.hadm_id, adm.disch_dt, next_adm.next_admit_dt,
    pat.gender, pat.anchor_age, adm.admission_type, adm.insurance,
    did.long_title, lab.max_creatinine, lab.min_hemoglobin, vf.avg_heart_rate;

