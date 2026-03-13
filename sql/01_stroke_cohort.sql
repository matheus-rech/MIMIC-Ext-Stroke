-- Extract ischemic stroke patients with ICU stays from MIMIC-IV v3.1
-- Reference: Abdollahi et al. (2025) cohort definition

WITH stroke_diagnoses AS (
    SELECT DISTINCT
        d.subject_id,
        d.hadm_id,
        d.icd_code,
        d.icd_version,
        d.seq_num,
        diag.long_title AS icd_title
    FROM read_csv_auto('{mimic_path}/hosp/diagnoses_icd.csv.gz') d
    JOIN read_csv_auto('{mimic_path}/hosp/d_icd_diagnoses.csv.gz') diag
        ON d.icd_code = diag.icd_code AND d.icd_version = diag.icd_version
    WHERE
        (d.icd_version = 9 AND (
            d.icd_code LIKE '433%' OR
            d.icd_code LIKE '434%' OR
            d.icd_code LIKE '435%' OR
            d.icd_code = '436'
        ))
        OR
        (d.icd_version = 10 AND (
            d.icd_code LIKE 'I60%' OR
            d.icd_code LIKE 'I61%' OR
            d.icd_code LIKE 'I63%' OR
            d.icd_code LIKE 'I65%' OR
            d.icd_code LIKE 'I66%' OR
            d.icd_code LIKE 'I67%' OR
            d.icd_code LIKE 'G45%'
        ))
),

stroke_icu AS (
    SELECT
        sd.subject_id,
        sd.hadm_id,
        sd.icd_code,
        sd.icd_version,
        sd.seq_num,
        sd.icd_title,
        i.stay_id,
        i.first_careunit,
        i.last_careunit,
        i.intime,
        i.outtime,
        i.los,
        p.gender,
        p.anchor_age,
        p.dod,
        a.admittime,
        a.dischtime,
        a.deathtime,
        a.admission_type,
        a.admission_location,
        a.discharge_location,
        a.insurance,
        a.race,
        a.hospital_expire_flag,
        ROW_NUMBER() OVER (
            PARTITION BY sd.subject_id
            ORDER BY i.intime ASC
        ) AS icu_stay_rank
    FROM stroke_diagnoses sd
    JOIN read_csv_auto('{mimic_path}/icu/icustays.csv.gz') i
        ON sd.subject_id = i.subject_id AND sd.hadm_id = i.hadm_id
    JOIN read_csv_auto('{mimic_path}/hosp/patients.csv.gz') p
        ON sd.subject_id = p.subject_id
    JOIN read_csv_auto('{mimic_path}/hosp/admissions.csv.gz') a
        ON sd.hadm_id = a.hadm_id
    WHERE i.los >= {min_icu_los_hours} / 24.0
      AND i.los <= {max_icu_los_days}
)

SELECT * EXCLUDE (icu_stay_rank)
FROM stroke_icu
WHERE icu_stay_rank = 1
ORDER BY subject_id;
