-- Static feature extraction: comorbidities, stroke subtype, first-24h labs
-- Uses cohort parquet as base, joins diagnoses and labs from MIMIC-IV

WITH cohort AS (
    SELECT *
    FROM read_parquet('{cohort_path}/stroke_cohort.parquet')
),

-- Get ALL diagnoses for cohort patients (not just stroke dx)
all_diagnoses AS (
    SELECT
        d.subject_id,
        d.hadm_id,
        d.icd_code,
        d.icd_version
    FROM read_csv_auto('{mimic_path}/hosp/diagnoses_icd.csv.gz') d
    WHERE d.subject_id IN (SELECT DISTINCT subject_id FROM cohort)
),

-- Comorbidity flags per patient (any admission, not just index)
comorbidities AS (
    SELECT
        c.subject_id,
        c.hadm_id,
        MAX(CASE WHEN
            (d.icd_version = 9 AND d.icd_code LIKE '401%') OR
            (d.icd_version = 10 AND d.icd_code LIKE 'I10%')
            THEN 1 ELSE 0 END) AS has_hypertension,
        MAX(CASE WHEN
            (d.icd_version = 9 AND d.icd_code LIKE '250%') OR
            (d.icd_version = 10 AND d.icd_code LIKE 'E11%')
            THEN 1 ELSE 0 END) AS has_diabetes,
        MAX(CASE WHEN
            (d.icd_version = 9 AND d.icd_code LIKE '4273%') OR
            (d.icd_version = 10 AND d.icd_code LIKE 'I48%')
            THEN 1 ELSE 0 END) AS has_afib,
        MAX(CASE WHEN
            (d.icd_version = 9 AND d.icd_code LIKE '272%') OR
            (d.icd_version = 10 AND d.icd_code LIKE 'E78%')
            THEN 1 ELSE 0 END) AS has_dyslipidemia,
        MAX(CASE WHEN
            (d.icd_version = 9 AND d.icd_code LIKE '585%') OR
            (d.icd_version = 10 AND d.icd_code LIKE 'N18%')
            THEN 1 ELSE 0 END) AS has_ckd,
        MAX(CASE WHEN
            (d.icd_version = 9 AND (d.icd_code LIKE '410%' OR d.icd_code LIKE '412%')) OR
            (d.icd_version = 10 AND (d.icd_code LIKE 'I21%' OR d.icd_code LIKE 'I25%'))
            THEN 1 ELSE 0 END) AS has_cad
    FROM cohort c
    LEFT JOIN all_diagnoses d
        ON c.subject_id = d.subject_id AND c.hadm_id = d.hadm_id
    GROUP BY c.subject_id, c.hadm_id
),

-- Stroke subtype from index admission ICD code
stroke_subtype AS (
    SELECT
        c.subject_id,
        c.hadm_id,
        CASE
            WHEN c.icd_code LIKE 'I63%' THEN 'ischemic'
            WHEN c.icd_code LIKE 'I61%' THEN 'ich'
            WHEN c.icd_code LIKE 'I60%' THEN 'sah'
            WHEN c.icd_code LIKE 'G45%' THEN 'tia'
            WHEN c.icd_version = 9 AND (c.icd_code LIKE '433%' OR c.icd_code LIKE '434%') THEN 'ischemic'
            WHEN c.icd_version = 9 AND c.icd_code LIKE '435%' THEN 'tia'
            ELSE 'other'
        END AS stroke_subtype
    FROM cohort c
),

-- First-24h admission labs (filter early by subject_id and itemid)
cohort_labs_raw AS (
    SELECT
        l.subject_id,
        l.hadm_id,
        l.itemid,
        l.valuenum,
        l.charttime
    FROM read_csv_auto('{mimic_path}/hosp/labevents.csv.gz') l
    WHERE l.subject_id IN (SELECT DISTINCT subject_id FROM cohort)
      AND l.itemid IN (50931, 50983, 50912, 51222, 51265, 51237)
      AND l.valuenum IS NOT NULL
),

-- Filter to first 24h and pick first value per lab
first_24h_labs AS (
    SELECT
        lr.subject_id,
        lr.hadm_id,
        lr.itemid,
        lr.valuenum,
        ROW_NUMBER() OVER (
            PARTITION BY lr.subject_id, lr.hadm_id, lr.itemid
            ORDER BY lr.charttime ASC
        ) AS rn
    FROM cohort_labs_raw lr
    JOIN cohort c
        ON lr.subject_id = c.subject_id AND lr.hadm_id = c.hadm_id
    WHERE lr.charttime >= c.admittime
      AND lr.charttime <= c.admittime + INTERVAL '24 hours'
),

labs_pivoted AS (
    SELECT
        subject_id,
        hadm_id,
        MAX(CASE WHEN itemid = 50931 THEN valuenum END) AS lab_glucose,
        MAX(CASE WHEN itemid = 50983 THEN valuenum END) AS lab_sodium,
        MAX(CASE WHEN itemid = 50912 THEN valuenum END) AS lab_creatinine,
        MAX(CASE WHEN itemid = 51222 THEN valuenum END) AS lab_hemoglobin,
        MAX(CASE WHEN itemid = 51265 THEN valuenum END) AS lab_platelets,
        MAX(CASE WHEN itemid = 51237 THEN valuenum END) AS lab_inr
    FROM first_24h_labs
    WHERE rn = 1
    GROUP BY subject_id, hadm_id
)

SELECT
    c.*,
    cm.has_hypertension,
    cm.has_diabetes,
    cm.has_afib,
    cm.has_dyslipidemia,
    cm.has_ckd,
    cm.has_cad,
    st.stroke_subtype,
    lp.lab_glucose,
    lp.lab_sodium,
    lp.lab_creatinine,
    lp.lab_hemoglobin,
    lp.lab_platelets,
    lp.lab_inr
FROM cohort c
LEFT JOIN comorbidities cm
    ON c.subject_id = cm.subject_id AND c.hadm_id = cm.hadm_id
LEFT JOIN stroke_subtype st
    ON c.subject_id = st.subject_id AND c.hadm_id = st.hadm_id
LEFT JOIN labs_pivoted lp
    ON c.subject_id = lp.subject_id AND c.hadm_id = lp.hadm_id
ORDER BY c.subject_id;
