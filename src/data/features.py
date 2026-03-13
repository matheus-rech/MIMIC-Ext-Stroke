"""Feature extraction for stroke digital twin."""
import duckdb
import pandas as pd
from pathlib import Path


def extract_static_features(config: dict, cohort: pd.DataFrame) -> pd.DataFrame:
    """Extract static features (comorbidities, labs, stroke subtype) for cohort.

    Parameters
    ----------
    config : dict
        Project configuration with data paths.
    cohort : pd.DataFrame
        Cohort dataframe with subject_id, hadm_id, admittime, etc.

    Returns
    -------
    pd.DataFrame
        One row per patient with demographics, comorbidities, stroke subtype,
        and first-24h admission lab values.
    """
    mimic_path = Path(config["data"]["mimic_path"]).resolve()
    cohort_path = Path(config["data"]["cohort_path"]).resolve()
    sql_path = Path("sql/02_static_features.sql")

    sql = sql_path.read_text()
    sql = sql.replace("{mimic_path}", str(mimic_path))
    sql = sql.replace("{cohort_path}", str(cohort_path))

    con = duckdb.connect()
    df = con.execute(sql).fetchdf()
    con.close()

    out_path = Path(config["data"]["cohort_path"])
    out_path.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path / "static_features.parquet", index=False)

    return df


def extract_timeseries(config: dict, cohort: pd.DataFrame) -> pd.DataFrame:
    """Extract hourly ICU time-series (vitals, GCS) for stroke cohort.

    Reads chartevents from MIMIC-IV, filters to cohort stay_ids and
    relevant vital/GCS itemids, computes hourly medians, and pivots
    to wide format.

    Parameters
    ----------
    config : dict
        Project configuration with data paths and timeseries settings.
    cohort : pd.DataFrame
        Cohort dataframe with stay_id, intime, outtime.

    Returns
    -------
    pd.DataFrame
        Long-format dataframe: (subject_id, stay_id, hour, hr, sbp, ...).
    """
    mimic_path = Path(config["data"]["mimic_path"]).resolve()
    cohort_path = Path(config["data"]["cohort_path"]).resolve()
    ts_cfg = config["timeseries"]
    sql_path = Path("sql/03_timeseries_features.sql")

    sql = sql_path.read_text()
    sql = sql.replace("{mimic_path}", str(mimic_path))
    sql = sql.replace("{cohort_path}", str(cohort_path))
    sql = sql.replace("{max_hours}", str(ts_cfg["max_hours"]))

    con = duckdb.connect()
    df = con.execute(sql).fetchdf()
    con.close()

    # Save long-format parquet
    out_path = Path(config["data"]["cohort_path"])
    out_path.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path / "timeseries.parquet", index=False)

    return df
