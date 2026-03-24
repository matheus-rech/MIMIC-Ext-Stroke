-- 04_eicu_stroke_cohort.sql
-- Extract stroke cohort from eICU-CRD 2.0 CSV.GZ files via DuckDB.
-- Matches stroke patients by free-text diagnosisstring patterns,
-- harmonises columns to MIMIC-IV schema for cross-dataset comparison.

-- ============================================================
-- 1. Patient demographics (adults only)
-- ============================================================
WITH patients AS (
    SELECT
        p.patientunitstayid,
        p.patienthealthsystemstayid,
        p.gender,
        CASE
            WHEN p.age = '> 89' THEN 90
            WHEN TRY_CAST(p.age AS INTEGER) IS NOT NULL THEN CAST(p.age AS INTEGER)
            ELSE NULL
        END AS age,
        p.ethnicity,
        p.hospitalid,
        p.unittype,
        -- eICU offsets are relative to ICU admission, so unitadmitoffset = 0
        0 AS unitadmitoffset,
        p.unitdischargeoffset,
        p.hospitaldischargestatus
    FROM read_csv_auto('{eicu_path}/patient.csv.gz', header=true) p
    WHERE (
        TRY_CAST(p.age AS INTEGER) >= 18
        OR p.age = '> 89'
    )
),

-- ============================================================
-- 2. Stroke diagnoses via free-text pattern matching
-- ============================================================
stroke_dx AS (
    SELECT DISTINCT
        d.patientunitstayid,
        d.diagnosisoffset,
        d.diagnosisstring,
        d.icd9code,
        -- Classify stroke subtype from free-text
        CASE
            WHEN LOWER(d.diagnosisstring) LIKE '%subarachnoid hemorrhage%'
                THEN 'sah'
            WHEN LOWER(d.diagnosisstring) LIKE '%intracerebral hemorrhage%'
              OR LOWER(d.diagnosisstring) LIKE '%intracranial hemorrhage%'
                THEN 'ich'
            WHEN LOWER(d.diagnosisstring) LIKE '%transient ischemic%'
                THEN 'tia'
            WHEN LOWER(d.diagnosisstring) LIKE '%ischemic%'
              OR LOWER(d.diagnosisstring) LIKE '%infarction%'
              OR LOWER(d.diagnosisstring) LIKE '%thrombosis%'
              OR LOWER(d.diagnosisstring) LIKE '%embolism%'
                THEN 'ischemic'
            ELSE 'other'
        END AS stroke_subtype
    FROM read_csv_auto('{eicu_path}/diagnosis.csv.gz', header=true) d
    WHERE LOWER(d.diagnosisstring) LIKE '%stroke%'
       OR LOWER(d.diagnosisstring) LIKE '%cerebrovascular accident%'
       OR LOWER(d.diagnosisstring) LIKE '% cva %'
       OR LOWER(d.diagnosisstring) LIKE '% cva,%'
       OR LOWER(d.diagnosisstring) LIKE '%cerebral infarction%'
       OR LOWER(d.diagnosisstring) LIKE '%ischemic stroke%'
       OR LOWER(d.diagnosisstring) LIKE '%intracerebral hemorrhage%'
       OR LOWER(d.diagnosisstring) LIKE '%intracranial hemorrhage%'
       OR LOWER(d.diagnosisstring) LIKE '%subarachnoid hemorrhage%'
       OR LOWER(d.diagnosisstring) LIKE '%transient ischemic%'
       OR LOWER(d.diagnosisstring) LIKE '%cerebral embolism%'
       OR LOWER(d.diagnosisstring) LIKE '%cerebral thrombosis%'
       OR LOWER(d.diagnosisstring) LIKE '%brain infarction%'
),

-- ============================================================
-- 3. Mortality from APACHE patient result
-- ============================================================
mortality AS (
    SELECT
        ar.patientunitstayid,
        ar.actualhospitalmortality,
        ar.apachescore
    FROM (
        SELECT
            ar2.patientunitstayid,
            ar2.actualhospitalmortality,
            ar2.apachescore,
            ROW_NUMBER() OVER (
                PARTITION BY ar2.patientunitstayid
                ORDER BY ar2.apachescore DESC NULLS LAST
            ) AS rn
        FROM read_csv_auto('{eicu_path}/apachePatientResult.csv.gz', header=true) ar2
    ) ar
    WHERE ar.rn = 1
),

-- ============================================================
-- 4. GCS components from APACHE APS variables
-- ============================================================
gcs_apache AS (
    SELECT
        a.patientunitstayid,
        a.eyes   AS gcs_eye,
        a.motor  AS gcs_motor,
        a.verbal AS gcs_verbal,
        CASE
            WHEN a.eyes IS NOT NULL AND a.motor IS NOT NULL AND a.verbal IS NOT NULL
            THEN a.eyes + a.motor + a.verbal
            ELSE NULL
        END AS gcs_total
    FROM read_csv_auto('{eicu_path}/apacheapsvar.csv.gz', header=true) a
),

-- ============================================================
-- 5. Join: stroke patients with demographics, mortality, GCS
-- ============================================================
stroke_cohort_raw AS (
    SELECT
        p.patientunitstayid,
        p.patienthealthsystemstayid,
        p.gender,
        p.age,
        p.ethnicity,
        p.hospitalid,
        p.unittype,
        p.unitadmitoffset,
        p.unitdischargeoffset,
        p.hospitaldischargestatus,
        -- ICU length of stay in days (offsets in minutes)
        ROUND((p.unitdischargeoffset - p.unitadmitoffset) / 1440.0, 2) AS los,
        sd.diagnosisstring,
        sd.stroke_subtype,
        sd.icd9code,
        m.actualhospitalmortality,
        m.apachescore,
        g.gcs_eye,
        g.gcs_motor,
        g.gcs_verbal,
        g.gcs_total,
        CASE
            WHEN LOWER(m.actualhospitalmortality) = 'expired' THEN 1
            ELSE 0
        END AS hospital_expire_flag,
        ROW_NUMBER() OVER (
            PARTITION BY p.patienthealthsystemstayid
            ORDER BY p.unitadmitoffset ASC
        ) AS stay_rank
    FROM patients p
    INNER JOIN stroke_dx sd
        ON p.patientunitstayid = sd.patientunitstayid
    LEFT JOIN mortality m
        ON p.patientunitstayid = m.patientunitstayid
    LEFT JOIN gcs_apache g
        ON p.patientunitstayid = g.patientunitstayid
    WHERE (p.unitdischargeoffset - p.unitadmitoffset) >= {min_icu_los_hours} * 60
      AND (p.unitdischargeoffset - p.unitadmitoffset) <= {max_icu_los_days} * 1440
)

-- Keep first ICU stay per patient (matches MIMIC cohort strategy)
SELECT * EXCLUDE (stay_rank)
FROM stroke_cohort_raw
WHERE stay_rank = 1
ORDER BY patienthealthsystemstayid;
