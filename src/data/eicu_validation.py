"""External validation of stroke digital twin using eICU-CRD 2.0.

Extracts a stroke cohort from eICU CSV.GZ files via DuckDB, harmonises
column names to the MIMIC-IV schema produced by ``extract.py``, and
provides cohort-level demographic comparison utilities.

eICU-CRD 2.0 reference: https://eicu-crd.mit.edu/
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stroke diagnosis patterns for eICU free-text ``diagnosisstring``
# ---------------------------------------------------------------------------
STROKE_DIAGNOSIS_PATTERNS: list[str] = [
    "%stroke%",
    "%cerebrovascular accident%",
    "%CVA%",
    "%cerebral infarction%",
    "%ischemic stroke%",
    "%intracerebral hemorrhage%",
    "%intracranial hemorrhage%",
    "%subarachnoid hemorrhage%",
    "%SAH%",
    "%transient ischemic attack%",
    "%TIA%",
    "%cerebral embolism%",
    "%cerebral thrombosis%",
    "%brain infarction%",
]

# Vital-sign column names shared with MIMIC timeseries schema
TIMESERIES_COLUMNS: list[str] = [
    "hr",
    "sbp",
    "dbp",
    "map",
    "rr",
    "spo2",
    "temp_c",
    "gcs_eye",
    "gcs_verbal",
    "gcs_motor",
    "gcs_total",
]


# ===================================================================
# 1.  Cohort extraction
# ===================================================================

def extract_eicu_stroke_cohort(config: dict) -> pd.DataFrame:
    """Extract stroke ICU cohort from eICU-CRD 2.0 CSV.GZ files via DuckDB.

    Uses ``sql/04_eicu_stroke_cohort.sql`` with free-text diagnosis
    matching.  The output schema is harmonised to be comparable with
    the MIMIC-IV cohort produced by :func:`extract.extract_stroke_cohort`.

    Parameters
    ----------
    config : dict
        Configuration dict.  Must contain:

        - ``data.eicu_path``: directory containing eICU CSV.GZ files
        - ``data.eicu_output_path``: output directory for eICU parquets
        - ``cohort.min_icu_los_hours``, ``cohort.max_icu_los_days``

    Returns
    -------
    pd.DataFrame
        One row per patient (first ICU stay), harmonised to MIMIC-like
        columns: ``subject_id``, ``hadm_id``, ``gender``, ``anchor_age``,
        ``los``, ``hospital_expire_flag``, ``stroke_subtype``, etc.
    """
    eicu_path = Path(config["data"]["eicu_path"]).resolve()
    cohort_cfg = config["cohort"]
    sql_path = Path(__file__).resolve().parent.parent.parent / "sql" / "04_eicu_stroke_cohort.sql"

    sql = sql_path.read_text()
    sql = sql.replace("{eicu_path}", str(eicu_path))
    sql = sql.replace("{min_icu_los_hours}", str(cohort_cfg["min_icu_los_hours"]))
    sql = sql.replace("{max_icu_los_days}", str(cohort_cfg["max_icu_los_days"]))

    logger.info("Running eICU stroke cohort SQL on %s ...", eicu_path)
    con = duckdb.connect()
    df = con.execute(sql).fetchdf()
    con.close()
    logger.info("Raw eICU stroke cohort: %d patients", len(df))

    # --- harmonise column names to MIMIC schema ---
    df = _harmonise_eicu_cohort(df)

    # --- persist ---
    out_path = Path(config["data"].get("eicu_output_path", "outputs/eicu_validation"))
    out_path.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path / "eicu_stroke_cohort.parquet", index=False)
    logger.info(
        "eICU stroke cohort saved (%d patients) -> %s",
        len(df),
        out_path / "eicu_stroke_cohort.parquet",
    )

    return df


def _harmonise_eicu_cohort(df: pd.DataFrame) -> pd.DataFrame:
    """Map eICU column names to the MIMIC-IV schema used elsewhere."""
    rename_map = {
        "patienthealthsystemstayid": "subject_id",
        "patientunitstayid": "hadm_id",
        "age": "anchor_age",
        "hospitaldischargestatus": "discharge_status",
        "actualhospitalmortality": "mortality_label",
    }
    df = df.rename(columns=rename_map)

    # Gender harmonisation: eICU uses "Male"/"Female"; MIMIC uses "M"/"F"
    if "gender" in df.columns:
        df["gender"] = df["gender"].map(
            {"Male": "M", "Female": "F", "male": "M", "female": "F"}
        ).fillna(df["gender"])

    # Ethnicity -> race (MIMIC nomenclature)
    if "ethnicity" in df.columns:
        df = df.rename(columns={"ethnicity": "race"})

    # Ensure hospital_expire_flag is int
    if "hospital_expire_flag" in df.columns:
        df["hospital_expire_flag"] = df["hospital_expire_flag"].astype(int)

    # Add provenance marker
    df["data_source"] = "eICU-CRD 2.0"

    return df


# ===================================================================
# 2.  Time-series extraction (72 h vitals + GCS)
# ===================================================================

def extract_eicu_stroke_timeseries(
    config: dict,
    cohort: pd.DataFrame,
) -> pd.DataFrame:
    """Extract 72-hour ICU time-series from eICU for the stroke cohort.

    Pulls vital signs from ``vitalperiodic.csv.gz`` and GCS components
    from ``apacheapsvar.csv.gz``, resamples to hourly medians, and
    returns a long-format DataFrame matching the MIMIC timeseries schema
    (columns: ``subject_id``, ``stay_id``, ``hour``, ``hr``, ``sbp``, ...).

    Parameters
    ----------
    config : dict
        Must contain ``data.eicu_path`` and ``timeseries.max_hours``.
    cohort : pd.DataFrame
        eICU stroke cohort (output of :func:`extract_eicu_stroke_cohort`).
        Must have ``hadm_id`` (= ``patientunitstayid``) and ``subject_id``.

    Returns
    -------
    pd.DataFrame
        Hourly time-series with MIMIC-compatible column names.
    """
    eicu_path = Path(config["data"]["eicu_path"]).resolve()
    max_hours = int(config.get("timeseries", {}).get("max_hours", 72))
    stay_ids = cohort["hadm_id"].unique().tolist()

    if not stay_ids:
        logger.warning("Empty cohort -- returning empty timeseries DataFrame.")
        return pd.DataFrame(columns=["subject_id", "stay_id", "hour"] + TIMESERIES_COLUMNS)

    # --- Map hadm_id -> subject_id for later join ---
    id_map = cohort[["hadm_id", "subject_id"]].drop_duplicates().set_index("hadm_id")["subject_id"]

    con = duckdb.connect()

    # ---- 2a.  Vital signs from vitalperiodic ----
    logger.info("Reading eICU vitalperiodic for %d stays ...", len(stay_ids))
    vitals_sql = f"""
    SELECT
        v.patientunitstayid,
        FLOOR(v.observationoffset / 60)::INTEGER AS hour,
        MEDIAN(v.heartrate)       AS hr,
        MEDIAN(v.systemicsystolic) AS sbp,
        MEDIAN(v.systemicdiastolic) AS dbp,
        MEDIAN(v.systemicmean)    AS map,
        MEDIAN(v.respiration)     AS rr,
        MEDIAN(v.sao2)            AS spo2,
        MEDIAN(v.temperature)     AS temp_c
    FROM read_csv_auto('{eicu_path}/vitalPeriodic.csv.gz', header=true) v
    WHERE v.patientunitstayid IN ({','.join(str(s) for s in stay_ids)})
      AND v.observationoffset >= 0
      AND FLOOR(v.observationoffset / 60) <= {max_hours}
    GROUP BY v.patientunitstayid, FLOOR(v.observationoffset / 60)::INTEGER
    """
    vitals = con.execute(vitals_sql).fetchdf()
    logger.info("Extracted %d hourly vital rows from eICU.", len(vitals))

    # Temperature: eICU stores Fahrenheit; convert to Celsius
    if "temp_c" in vitals.columns:
        mask = vitals["temp_c"] > 50  # clearly Fahrenheit
        vitals.loc[mask, "temp_c"] = (vitals.loc[mask, "temp_c"] - 32.0) * 5.0 / 9.0

    vitals = vitals.rename(columns={"patientunitstayid": "stay_id"})

    # ---- 2b.  GCS from apacheapsvar (one value per stay) ----
    logger.info("Reading eICU apacheapsvar for GCS ...")
    gcs_sql = f"""
    SELECT
        a.patientunitstayid,
        a.eyes   AS gcs_eye,
        a.motor  AS gcs_motor,
        a.verbal AS gcs_verbal
    FROM read_csv_auto('{eicu_path}/apacheApsVar.csv.gz', header=true) a
    WHERE a.patientunitstayid IN ({','.join(str(s) for s in stay_ids)})
    """
    gcs = con.execute(gcs_sql).fetchdf()
    gcs = gcs.rename(columns={"patientunitstayid": "stay_id"})
    if not gcs.empty:
        gcs["gcs_total"] = gcs[["gcs_eye", "gcs_motor", "gcs_verbal"]].sum(
            axis=1, skipna=False
        )
    logger.info("Extracted GCS for %d stays.", len(gcs))

    con.close()

    # ---- 2c.  Merge vitals + GCS ----
    if not gcs.empty:
        # Broadcast single GCS value across all hours for each stay
        ts = vitals.merge(gcs, on="stay_id", how="left")
    else:
        ts = vitals.copy()
        for col in ["gcs_eye", "gcs_verbal", "gcs_motor", "gcs_total"]:
            ts[col] = np.nan

    # Map stay_id -> subject_id
    ts["subject_id"] = ts["stay_id"].map(id_map)

    # Reorder to match MIMIC schema
    out_cols = ["subject_id", "stay_id", "hour"] + TIMESERIES_COLUMNS
    for col in out_cols:
        if col not in ts.columns:
            ts[col] = np.nan
    ts = ts[out_cols].sort_values(["stay_id", "hour"]).reset_index(drop=True)

    # ---- persist ----
    out_path = Path(config["data"].get("eicu_output_path", "outputs/eicu_validation"))
    out_path.mkdir(parents=True, exist_ok=True)
    ts.to_parquet(out_path / "eicu_timeseries.parquet", index=False)
    logger.info(
        "eICU timeseries saved (%d rows, %d stays) -> %s",
        len(ts),
        ts["stay_id"].nunique(),
        out_path / "eicu_timeseries.parquet",
    )

    return ts


# ===================================================================
# 3.  Demographic comparison (MIMIC vs eICU)
# ===================================================================

def compare_cohort_demographics(
    mimic_cohort: pd.DataFrame,
    eicu_cohort: pd.DataFrame,
) -> pd.DataFrame:
    """Compare demographics between MIMIC-IV and eICU stroke cohorts.

    Computes summary statistics for age, gender distribution, stroke
    subtype distribution, mortality rate, and ICU length of stay, then
    returns a side-by-side comparison table.

    Parameters
    ----------
    mimic_cohort : pd.DataFrame
        MIMIC stroke cohort (from ``extract_stroke_cohort``).
    eicu_cohort : pd.DataFrame
        eICU stroke cohort (from ``extract_eicu_stroke_cohort``).

    Returns
    -------
    pd.DataFrame
        Comparison table with rows = metrics, columns = [MIMIC-IV, eICU, p_value].
    """

    def _summarise(df: pd.DataFrame, label: str) -> dict:
        age_col = "anchor_age" if "anchor_age" in df.columns else "age"
        los_col = "los"
        gender_col = "gender"
        mortality_col = "hospital_expire_flag"
        subtype_col = "stroke_subtype"

        n = len(df)
        stats = {"source": label, "n_patients": n}

        # Age
        if age_col in df.columns:
            age = pd.to_numeric(df[age_col], errors="coerce")
            stats["age_mean"] = round(age.mean(), 1)
            stats["age_std"] = round(age.std(), 1)
            stats["age_median"] = round(age.median(), 1)
        else:
            stats["age_mean"] = stats["age_std"] = stats["age_median"] = np.nan

        # Gender (% male)
        if gender_col in df.columns:
            male_frac = (df[gender_col].isin(["M", "Male"])).mean()
            stats["pct_male"] = round(male_frac * 100, 1)
        else:
            stats["pct_male"] = np.nan

        # Mortality
        if mortality_col in df.columns:
            stats["mortality_pct"] = round(df[mortality_col].mean() * 100, 1)
        else:
            stats["mortality_pct"] = np.nan

        # LOS
        if los_col in df.columns:
            los = pd.to_numeric(df[los_col], errors="coerce")
            stats["los_median_days"] = round(los.median(), 1)
            stats["los_iqr_25"] = round(los.quantile(0.25), 1)
            stats["los_iqr_75"] = round(los.quantile(0.75), 1)
        else:
            stats["los_median_days"] = stats["los_iqr_25"] = stats["los_iqr_75"] = np.nan

        # Stroke subtype distribution
        if subtype_col in df.columns:
            dist = df[subtype_col].value_counts(normalize=True) * 100
            for st in ["ischemic", "ich", "sah", "tia", "other"]:
                stats[f"subtype_{st}_pct"] = round(dist.get(st, 0.0), 1)
        else:
            for st in ["ischemic", "ich", "sah", "tia", "other"]:
                stats[f"subtype_{st}_pct"] = np.nan

        return stats

    mimic_stats = _summarise(mimic_cohort, "MIMIC-IV")
    eicu_stats = _summarise(eicu_cohort, "eICU-CRD")

    # --- Statistical tests ---
    from scipy import stats as sp_stats

    p_values: dict[str, float | str] = {}

    # Age: two-sample t-test (Welch)
    age_col_m = "anchor_age" if "anchor_age" in mimic_cohort.columns else "age"
    age_col_e = "anchor_age" if "anchor_age" in eicu_cohort.columns else "age"
    if age_col_m in mimic_cohort.columns and age_col_e in eicu_cohort.columns:
        m_age = pd.to_numeric(mimic_cohort[age_col_m], errors="coerce").dropna()
        e_age = pd.to_numeric(eicu_cohort[age_col_e], errors="coerce").dropna()
        if len(m_age) > 1 and len(e_age) > 1:
            _, p = sp_stats.ttest_ind(m_age, e_age, equal_var=False)
            p_values["age"] = round(p, 4)

    # Gender: chi-squared test
    if "gender" in mimic_cohort.columns and "gender" in eicu_cohort.columns:
        m_male = (mimic_cohort["gender"].isin(["M", "Male"])).sum()
        m_total = len(mimic_cohort)
        e_male = (eicu_cohort["gender"].isin(["M", "Male"])).sum()
        e_total = len(eicu_cohort)
        table = np.array([[m_male, m_total - m_male], [e_male, e_total - e_male]])
        if table.min() >= 0 and table.sum() > 0:
            _, p, _, _ = sp_stats.chi2_contingency(table)
            p_values["gender"] = round(p, 4)

    # Mortality: chi-squared test
    if "hospital_expire_flag" in mimic_cohort.columns and "hospital_expire_flag" in eicu_cohort.columns:
        m_dead = int(mimic_cohort["hospital_expire_flag"].sum())
        e_dead = int(eicu_cohort["hospital_expire_flag"].sum())
        table = np.array(
            [[m_dead, len(mimic_cohort) - m_dead], [e_dead, len(eicu_cohort) - e_dead]]
        )
        if table.min() >= 0 and table.sum() > 0:
            _, p, _, _ = sp_stats.chi2_contingency(table)
            p_values["mortality"] = round(p, 4)

    # LOS: Mann-Whitney U test
    if "los" in mimic_cohort.columns and "los" in eicu_cohort.columns:
        m_los = pd.to_numeric(mimic_cohort["los"], errors="coerce").dropna()
        e_los = pd.to_numeric(eicu_cohort["los"], errors="coerce").dropna()
        if len(m_los) > 0 and len(e_los) > 0:
            _, p = sp_stats.mannwhitneyu(m_los, e_los, alternative="two-sided")
            p_values["los"] = round(p, 4)

    # Build comparison DataFrame
    rows = []
    metric_keys = [
        k for k in mimic_stats if k not in ("source",)
    ]
    for key in metric_keys:
        row = {
            "metric": key,
            "MIMIC-IV": mimic_stats.get(key),
            "eICU-CRD": eicu_stats.get(key),
        }
        # Attach p-value where available
        for p_key, p_val in p_values.items():
            if p_key in key:
                row["p_value"] = p_val
                break
        rows.append(row)

    comparison = pd.DataFrame(rows)
    logger.info("Cohort comparison:\n%s", comparison.to_string(index=False))

    return comparison
