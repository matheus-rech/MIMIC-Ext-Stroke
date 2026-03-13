"""Tests for CTGAN and TVAE baseline synthesizers."""
import pytest
import pandas as pd


@pytest.fixture
def static_data():
    df = pd.read_parquet("outputs/cohort/static_features.parquet")
    # Use subset for speed
    return df.head(500)


def test_ctgan_generates_samples(static_data):
    from src.models.ctgan_baseline import StrokeCTGAN

    model = StrokeCTGAN(epochs=5)
    model.fit(static_data)
    synthetic = model.sample(n=50)
    assert len(synthetic) == 50
    assert "stroke_subtype" in synthetic.columns
    assert "has_hypertension" in synthetic.columns


def test_tvae_generates_samples(static_data):
    from src.models.ctgan_baseline import StrokeTVAE

    model = StrokeTVAE(epochs=5)
    model.fit(static_data)
    synthetic = model.sample(n=50)
    assert len(synthetic) == 50


def test_ctgan_preserves_column_schema(static_data):
    from src.models.ctgan_baseline import StrokeCTGAN

    model = StrokeCTGAN(epochs=5)
    # Select same features as BN for fair comparison
    feature_cols = [
        "anchor_age",
        "gender",
        "stroke_subtype",
        "hospital_expire_flag",
        "los",
        "has_hypertension",
        "has_diabetes",
        "has_afib",
        "has_dyslipidemia",
        "has_ckd",
        "has_cad",
    ]
    cols = [c for c in feature_cols if c in static_data.columns]
    data = static_data[cols]
    model.fit(data)
    synthetic = model.sample(n=50)
    assert set(synthetic.columns) == set(data.columns)
