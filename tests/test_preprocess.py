"""Tests for preprocessing pipeline."""
import pandas as pd
import numpy as np


def test_encode_categoricals():
    from src.data.preprocess import encode_categoricals
    df = pd.DataFrame({
        "gender": ["M", "F", "M"],
        "stroke_subtype": ["ischemic", "ich", "sah"],
        "insurance": ["Medicare", "Medicaid", "Other"],
    })
    encoded = encode_categoricals(df)
    assert "gender_M" in encoded.columns or "gender" not in encoded.columns
    assert len(encoded) == 3


def test_impute_missing_static():
    from src.data.preprocess import impute_missing_static
    df = pd.DataFrame({
        "anchor_age": [65, np.nan, 70],
        "glucose_admit": [120, np.nan, np.nan],
        "has_hypertension": [1, 0, 1],
    })
    imputed = impute_missing_static(df)
    assert imputed["anchor_age"].notna().all()
    assert imputed["glucose_admit"].notna().all()
    assert "glucose_admit_missing" in imputed.columns


def test_impute_missing_static_mean():
    from src.data.preprocess import impute_missing_static
    df = pd.DataFrame({
        "anchor_age": [60.0, np.nan, 80.0],
        "lab_glucose": [100.0, np.nan, 200.0],
    })
    imputed_mean = impute_missing_static(df, method="mean")
    imputed_median = impute_missing_static(df, method="median")
    # Mean of [60, 80] = 70; median of [60, 80] = 70 (same for 2 values)
    assert imputed_mean["anchor_age"].notna().all()
    assert imputed_median["anchor_age"].notna().all()
    # Lab columns should get missingness flags regardless of method
    assert "lab_glucose_missing" in imputed_mean.columns


def test_impute_missing_static_invalid_method():
    from src.data.preprocess import impute_missing_static
    import pytest
    df = pd.DataFrame({"anchor_age": [65, np.nan, 70]})
    with pytest.raises(ValueError, match="Unsupported imputation method"):
        impute_missing_static(df, method="invalid")


def test_normalize_numeric():
    from src.data.preprocess import normalize_numeric
    df = pd.DataFrame({"age": [20, 40, 60, 80], "los": [1, 2, 3, 4]})
    normalized, params = normalize_numeric(df, ["age", "los"])
    assert normalized["age"].min() >= -1.1  # approximate
    assert normalized["age"].max() <= 1.1
    assert "age" in params


def test_split_data():
    from src.data.preprocess import split_data
    df = pd.DataFrame({
        "subject_id": range(100),
        "hospital_expire_flag": [0] * 85 + [1] * 15,
    })
    train, val, test = split_data(df, test_size=0.2, val_size=0.1, seed=42)
    assert len(train) + len(val) + len(test) == 100
    # Check stratification maintained
    assert abs(train["hospital_expire_flag"].mean() - 0.15) < 0.05


def test_preprocess_timeseries_forward_fill():
    from src.data.preprocess import preprocess_timeseries
    ts = pd.DataFrame({
        "subject_id": [1] * 5,
        "stay_id": [100] * 5,
        "hour": [0, 1, 2, 3, 4],
        "hr": [80, np.nan, np.nan, 85, np.nan],
        "spo2": [98, 97, np.nan, np.nan, 96],
    })
    result = preprocess_timeseries(ts)
    # Forward-fill should propagate values
    assert result.loc[result["hour"] == 1, "hr"].iloc[0] == 80  # forward-filled
    assert result.loc[result["hour"] == 2, "hr"].iloc[0] == 80  # forward-filled


def test_preprocess_pipeline_outputs():
    from src.data.preprocess import preprocess_pipeline
    import yaml

    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    result = preprocess_pipeline(config)
    assert "static_train" in result
    assert "static_test" in result
    assert "ts_processed" in result
    assert len(result["static_train"]) > 0
