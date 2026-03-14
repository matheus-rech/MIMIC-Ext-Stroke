"""Tests for the hybrid digital twin pipeline (BN + DGAN)."""

import pandas as pd


def test_hybrid_pipeline_creates():
    from src.models.hybrid import HybridDigitalTwin

    pipeline = HybridDigitalTwin()
    assert pipeline is not None


def test_hybrid_generates_static_profiles():
    from src.models.hybrid import HybridDigitalTwin

    static = pd.read_parquet("outputs/cohort/static_features.parquet")
    pipeline = HybridDigitalTwin()
    pipeline.fit_static(static.head(500))
    profiles = pipeline.generate_static(n=50, seed=42)
    assert len(profiles) == 50
    assert "stroke_subtype" in profiles.columns


def test_hybrid_generates_complete_patients():
    from src.models.hybrid import HybridDigitalTwin

    # Use tiny data for testing
    static = pd.read_parquet("outputs/cohort/static_features.parquet").head(200)
    ts = pd.read_parquet("outputs/cohort/timeseries_processed.parquet")

    pipeline = HybridDigitalTwin(dgan_epochs=3, dgan_hidden_dim=32, dgan_noise_dim=8)
    pipeline.fit(static, ts)

    result = pipeline.generate(n_patients=20, seed=42)
    assert "static" in result
    assert "timeseries" in result
    assert len(result["static"]) == 20
    assert result["timeseries"].shape[0] == 20  # 20 patients


def test_hybrid_generate_multiple_datasets():
    from src.models.hybrid import HybridDigitalTwin

    static = pd.read_parquet("outputs/cohort/static_features.parquet").head(200)
    ts = pd.read_parquet("outputs/cohort/timeseries_processed.parquet")

    pipeline = HybridDigitalTwin(dgan_epochs=2, dgan_hidden_dim=32, dgan_noise_dim=8)
    pipeline.fit(static, ts)

    datasets = pipeline.generate_multiple_datasets(n_patients=10, n_datasets=3, base_seed=0)
    assert len(datasets) == 3
    for ds in datasets:
        assert len(ds["static"]) == 10
        assert ds["timeseries"].shape[0] == 10


def test_hybrid_timeseries_shape():
    """Generated time-series should have correct seq_len and feature dims."""
    from src.models.hybrid import HybridDigitalTwin

    static = pd.read_parquet("outputs/cohort/static_features.parquet").head(200)
    ts = pd.read_parquet("outputs/cohort/timeseries_processed.parquet")

    seq_len = 48
    pipeline = HybridDigitalTwin(
        dgan_epochs=2, dgan_hidden_dim=32, dgan_noise_dim=8, seq_len=seq_len
    )
    pipeline.fit(static, ts)

    result = pipeline.generate(n_patients=5, seed=99)
    assert result["timeseries"].shape[1] == seq_len
    # Features should match the TS feature columns found during fit
    assert result["timeseries"].shape[2] == len(pipeline._ts_feature_cols)
