"""Extract stroke cohort from MIMIC-IV using DuckDB."""

import duckdb
import pandas as pd
from pathlib import Path


def extract_stroke_cohort(config: dict) -> pd.DataFrame:
    """Extract stroke ICU cohort from MIMIC-IV CSV.GZ files via DuckDB.

    Identifies stroke patients by ICD-9/10 codes, joins with ICU stays,
    keeps first ICU stay per patient. Based on Abdollahi et al. (2025).

    Args:
        config: Configuration dict with data paths and cohort parameters.

    Returns:
        DataFrame with one row per patient (first ICU stay).
    """
    mimic_path = Path(config["data"]["mimic_path"]).resolve()
    cohort_cfg = config["cohort"]
    sql_path = Path(__file__).resolve().parent.parent.parent / "sql" / "01_stroke_cohort.sql"

    sql = sql_path.read_text()
    sql = sql.replace("{mimic_path}", str(mimic_path))
    sql = sql.replace("{min_icu_los_hours}", str(cohort_cfg["min_icu_los_hours"]))
    sql = sql.replace("{max_icu_los_days}", str(cohort_cfg["max_icu_los_days"]))

    con = duckdb.connect()
    df = con.execute(sql).fetchdf()
    con.close()

    out_path = Path(config["data"]["cohort_path"])
    out_path.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path / "stroke_cohort.parquet", index=False)

    return df
