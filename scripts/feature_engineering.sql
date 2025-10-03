-- This query is upgraded to include a powerful, final "interaction-feature".

-- CTE for lab features
WITH lab_features AS (
    SELECT
        hadm_id,
        MAX(CASE WHEN itemid = 50912 THEN valuenum ELSE NULL END) AS max_creatinine,
        MIN(CASE WHEN itemid = 51222 THEN valuenum ELSE NULL END) AS min_hemoglobin
    FROM
        labevents
    GROUP BY
        hadm_id
),

-- CTE for vital signs
vital_features AS (
    SELECT
        adm.hadm_id,
        AVG(CASE WHEN ce.itemid IN (220045, 211) THEN ce.valuenum ELSE NULL END) AS avg_heart_rate
    FROM
        admissions adm
    INNER JOIN
        chartevents ce ON adm.hadm_id = ce.hadm_id
    WHERE
        ce.charttime BETWEEN adm.admit_dt AND adm.admit_dt + INTERVAL '1 day'
    GROUP BY
        adm.hadm_id
)

-- Final SELECT statement joining all features
SELECT
    adm.subject_id,
    adm.hadm_id,
    
    -- Target Variable
    EXTRACT(EPOCH FROM (adm.disch_dt - adm.admit_dt)) / 86400.0 AS length_of_stay_days,
    
    -- Demographics
    pat.gender,
    pat.anchor_age,
    
    -- Admission Details
    adm.admission_type,
    adm.insurance,
    
    -- Primary Diagnosis
    diag.icd_code AS primary_diagnosis,
    
    -- Procedure Features
    COUNT(DISTINCT proc.icd_code) AS procedure_count,
    
    -- Lab Features
    lab.max_creatinine,
    lab.min_hemoglobin,
    
    -- THE FINAL SUPER-FEATURE --
    -- This creates a risk-adjusted score by interacting age with the heart rate abnormality.
    -- A high value here is an unambiguous signal of high risk.
    (pat.anchor_age * CASE
            WHEN vf.avg_heart_rate IS NULL THEN 0
            ELSE ABS(vf.avg_heart_rate - 75) -- Calculate the deviation from a normal 75 bpm
        END
    ) AS age_hr_interaction
    
FROM
    admissions adm
INNER JOIN
    patients pat ON adm.subject_id = pat.subject_id
LEFT JOIN
    diagnoses_icd diag ON adm.hadm_id = diag.hadm_id AND diag.seq_num = 1
LEFT JOIN
    procedures_icd proc ON adm.hadm_id = proc.hadm_id
LEFT JOIN
    lab_features lab ON adm.hadm_id = lab.hadm_id
LEFT JOIN
    vital_features vf ON adm.hadm_id = vf.hadm_id
    
GROUP BY
    adm.subject_id,
    adm.hadm_id,
    adm.disch_dt,
    adm.admit_dt,
    pat.gender,
    pat.anchor_age,
    adm.admission_type,
    adm.insurance,
    diag.icd_code,
    lab.max_creatinine,
    lab.min_hemoglobin,
    vf.avg_heart_rate


