-- 03_timeseries_features.sql
-- Extract hourly-resampled ICU time-series (vitals + GCS) from chartevents
-- for the stroke cohort.

WITH cohort AS (
    SELECT stay_id, subject_id, intime, outtime
    FROM read_parquet('{cohort_path}/stroke_cohort.parquet')
),

-- Read chartevents filtered to relevant itemids and cohort stay_ids
raw_events AS (
    SELECT
        ce.subject_id,
        ce.stay_id,
        ce.charttime,
        ce.itemid,
        ce.valuenum,
        ce.value,
        c.intime
    FROM read_csv_auto('{mimic_path}/icu/chartevents.csv.gz', header=true, sample_size=100000) ce
    INNER JOIN cohort c ON ce.stay_id = c.stay_id
    WHERE ce.itemid IN (
        -- Vitals
        220045,  -- Heart Rate
        220050,  -- Arterial Blood Pressure systolic
        220051,  -- Arterial Blood Pressure diastolic
        220052,  -- Arterial Blood Pressure mean
        220179,  -- Non-Invasive Blood Pressure systolic
        220180,  -- Non-Invasive Blood Pressure diastolic
        220181,  -- Non-Invasive Blood Pressure mean
        220210,  -- Respiratory Rate
        220277,  -- SpO2
        223762,  -- Temperature Celsius
        223761,  -- Temperature Fahrenheit
        -- GCS
        220739,  -- GCS Eye Opening
        223900,  -- GCS Verbal Response
        223901   -- GCS Motor Response
    )
),

-- Compute hour offset and extract numeric value
events_with_hour AS (
    SELECT
        subject_id,
        stay_id,
        itemid,
        FLOOR(EXTRACT(EPOCH FROM (charttime - intime)) / 3600)::INTEGER AS hour,
        CASE
            -- Use valuenum if available
            WHEN valuenum IS NOT NULL THEN valuenum
            -- For GCS text values like "4 Spontaneously", extract leading number
            WHEN value IS NOT NULL AND regexp_matches(value, '^\d+')
                THEN CAST(regexp_extract(value, '^(\d+)', 1) AS DOUBLE)
            ELSE NULL
        END AS val
    FROM raw_events
    WHERE FLOOR(EXTRACT(EPOCH FROM (charttime - intime)) / 3600) >= 0
      AND FLOOR(EXTRACT(EPOCH FROM (charttime - intime)) / 3600) <= {max_hours}
),

-- Filter out null values
valid_events AS (
    SELECT * FROM events_with_hour WHERE val IS NOT NULL
),

-- Aggregate: median per (stay_id, hour, itemid)
hourly_median AS (
    SELECT
        subject_id,
        stay_id,
        hour,
        itemid,
        MEDIAN(val) AS median_val
    FROM valid_events
    GROUP BY subject_id, stay_id, hour, itemid
),

-- Pivot to wide format
pivoted AS (
    SELECT
        subject_id,
        stay_id,
        hour,
        MAX(CASE WHEN itemid = 220045 THEN median_val END) AS hr,
        -- Arterial BP (preferred)
        MAX(CASE WHEN itemid = 220050 THEN median_val END) AS sbp_art,
        MAX(CASE WHEN itemid = 220051 THEN median_val END) AS dbp_art,
        MAX(CASE WHEN itemid = 220052 THEN median_val END) AS map_art,
        -- Non-invasive BP (fallback)
        MAX(CASE WHEN itemid = 220179 THEN median_val END) AS sbp_ni,
        MAX(CASE WHEN itemid = 220180 THEN median_val END) AS dbp_ni,
        MAX(CASE WHEN itemid = 220181 THEN median_val END) AS map_ni,
        MAX(CASE WHEN itemid = 220210 THEN median_val END) AS rr,
        MAX(CASE WHEN itemid = 220277 THEN median_val END) AS spo2,
        MAX(CASE WHEN itemid = 223762 THEN median_val END) AS temp_c_direct,
        MAX(CASE WHEN itemid = 223761 THEN median_val END) AS temp_f,
        MAX(CASE WHEN itemid = 220739 THEN median_val END) AS gcs_eye,
        MAX(CASE WHEN itemid = 223900 THEN median_val END) AS gcs_verbal,
        MAX(CASE WHEN itemid = 223901 THEN median_val END) AS gcs_motor
    FROM hourly_median
    GROUP BY subject_id, stay_id, hour
)

SELECT
    subject_id,
    stay_id,
    hour,
    hr,
    -- Prefer arterial BP; fall back to non-invasive when arterial is absent
    COALESCE(sbp_art, sbp_ni) AS sbp,
    COALESCE(dbp_art, dbp_ni) AS dbp,
    COALESCE(map_art, map_ni) AS map,
    rr,
    spo2,
    -- Coalesce Celsius direct with converted Fahrenheit
    COALESCE(temp_c_direct, (temp_f - 32.0) * 5.0 / 9.0) AS temp_c,
    gcs_eye,
    gcs_verbal,
    gcs_motor,
    CASE
        WHEN gcs_eye IS NOT NULL AND gcs_verbal IS NOT NULL AND gcs_motor IS NOT NULL
        THEN gcs_eye + gcs_verbal + gcs_motor
        ELSE NULL
    END AS gcs_total
FROM pivoted
ORDER BY subject_id, stay_id, hour;
