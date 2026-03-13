import pytest
import numpy as np


def test_timeseries_returns_dataframe():
    from src.data.features import extract_timeseries
    import yaml, pandas as pd

    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    cohort = pd.read_parquet("outputs/cohort/stroke_cohort.parquet")
    ts = extract_timeseries(config, cohort)

    assert len(ts) > 0
    assert "subject_id" in ts.columns
    assert "stay_id" in ts.columns
    assert "hour" in ts.columns
    # Check vital sign columns
    for col in ["hr", "sbp", "dbp", "rr", "spo2", "temp_c", "gcs_eye", "gcs_verbal", "gcs_motor"]:
        assert col in ts.columns, f"Missing column: {col}"


def test_timeseries_hour_range():
    from src.data.features import extract_timeseries
    import yaml, pandas as pd

    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    cohort = pd.read_parquet("outputs/cohort/stroke_cohort.parquet")
    ts = extract_timeseries(config, cohort)

    # Hours should be 0 to max_hours (72)
    assert ts["hour"].min() >= 0
    assert ts["hour"].max() <= config["timeseries"]["max_hours"]


def test_timeseries_has_multiple_patients():
    from src.data.features import extract_timeseries
    import yaml, pandas as pd

    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    cohort = pd.read_parquet("outputs/cohort/stroke_cohort.parquet")
    ts = extract_timeseries(config, cohort)

    n_patients = ts["subject_id"].nunique()
    assert n_patients > 100, f"Only {n_patients} patients have time-series data"
