"""Tests for static feature extraction."""
import pytest


def test_static_features_shape():
    from src.data.features import extract_static_features
    import yaml, pandas as pd

    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    cohort = pd.read_parquet("outputs/cohort/stroke_cohort.parquet")
    static = extract_static_features(config, cohort)

    assert len(static) == len(cohort)
    assert "subject_id" in static.columns
    expected_cols = [
        "gender",
        "anchor_age",
        "hospital_expire_flag",
        "has_hypertension",
        "has_diabetes",
        "has_afib",
        "stroke_subtype",
        "los",
    ]
    for col in expected_cols:
        assert col in static.columns, f"Missing column: {col}"


def test_static_features_no_nulls_in_ids():
    from src.data.features import extract_static_features
    import yaml, pandas as pd

    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    cohort = pd.read_parquet("outputs/cohort/stroke_cohort.parquet")
    static = extract_static_features(config, cohort)
    assert static["subject_id"].notna().all()


def test_static_features_comorbidity_flags_are_boolean():
    from src.data.features import extract_static_features
    import yaml, pandas as pd

    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    cohort = pd.read_parquet("outputs/cohort/stroke_cohort.parquet")
    static = extract_static_features(config, cohort)

    comorbidity_cols = [
        "has_hypertension",
        "has_diabetes",
        "has_afib",
        "has_dyslipidemia",
        "has_ckd",
        "has_cad",
    ]
    for col in comorbidity_cols:
        assert col in static.columns, f"Missing comorbidity: {col}"
        assert static[col].isin([0, 1, True, False]).all(), f"{col} has non-boolean values"


def test_static_features_stroke_subtype_values():
    from src.data.features import extract_static_features
    import yaml, pandas as pd

    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    cohort = pd.read_parquet("outputs/cohort/stroke_cohort.parquet")
    static = extract_static_features(config, cohort)

    valid_subtypes = {"ischemic", "ich", "sah", "tia", "other"}
    assert set(static["stroke_subtype"].unique()).issubset(valid_subtypes)
