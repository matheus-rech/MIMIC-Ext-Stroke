"""End-to-end integration test for the stroke digital twin pipeline."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture(scope="module")
def config():
    import yaml
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)


class TestDataPipelineIntegration:
    """Test that data extraction and preprocessing produced valid outputs."""

    def test_cohort_exists(self):
        df = pd.read_parquet("outputs/cohort/stroke_cohort.parquet")
        assert len(df) > 1000
        assert df["subject_id"].is_unique

    def test_static_features_exist(self):
        df = pd.read_parquet("outputs/cohort/static_features.parquet")
        assert len(df) > 1000
        assert "stroke_subtype" in df.columns
        assert "has_hypertension" in df.columns

    def test_timeseries_exists(self):
        df = pd.read_parquet("outputs/cohort/timeseries.parquet")
        assert len(df) > 10000
        assert "hr" in df.columns

    def test_preprocessed_splits_exist(self):
        train = pd.read_parquet("outputs/cohort/static_features_train.parquet")
        val = pd.read_parquet("outputs/cohort/static_features_val.parquet")
        test = pd.read_parquet("outputs/cohort/static_features_test.parquet")
        assert len(train) > len(val)
        assert len(train) > len(test)


class TestModelIntegration:
    """Test that models can fit and generate on small subsets."""

    def test_bayesian_network_fit_and_sample(self):
        from src.models.bayesian_net import StrokeProfileBN
        static = pd.read_parquet("outputs/cohort/static_features.parquet").head(300)
        bn = StrokeProfileBN()
        bn.fit(static)
        synthetic = bn.sample(n=50, seed=42)
        assert len(synthetic) == 50
        assert "stroke_subtype" in synthetic.columns

    def test_dgan_fit_and_generate(self):
        from src.models.dgan_model import StrokeTimeSeriesDGAN
        n, t, f = 30, 24, 5
        metadata = np.random.randn(n, 3).astype(np.float32)
        sequences = np.random.randn(n, t, f).astype(np.float32)
        model = StrokeTimeSeriesDGAN(
            n_features=f, n_metadata=3, seq_len=t,
            noise_dim=8, hidden_dim=16, epochs=3, batch_size=4
        )
        model.train(metadata, sequences)
        generated = model.generate(np.random.randn(5, 3).astype(np.float32))
        assert generated.shape == (5, t, f)

    def test_ctgan_fit_and_sample(self):
        from src.models.ctgan_baseline import StrokeCTGAN
        static = pd.read_parquet("outputs/cohort/static_features.parquet").head(200)
        feature_cols = ["anchor_age", "gender", "stroke_subtype", "hospital_expire_flag",
                       "has_hypertension", "has_diabetes", "has_afib"]
        cols = [c for c in feature_cols if c in static.columns]
        model = StrokeCTGAN(epochs=3)
        model.fit(static[cols])
        synthetic = model.sample(n=20)
        assert len(synthetic) == 20

    def test_hybrid_pipeline(self):
        from src.models.hybrid import HybridDigitalTwin
        static = pd.read_parquet("outputs/cohort/static_features.parquet").head(100)
        ts = pd.read_parquet("outputs/cohort/timeseries_processed.parquet")

        pipeline = HybridDigitalTwin(dgan_epochs=3, dgan_hidden_dim=16, dgan_noise_dim=8)
        pipeline.fit(static, ts)
        result = pipeline.generate(n_patients=10, seed=42)
        assert len(result["static"]) == 10
        assert result["timeseries"].shape[0] == 10


class TestEvaluationIntegration:
    """Test that evaluation metrics work on real-ish data."""

    def test_fidelity_metrics(self):
        from src.evaluation.fidelity import dimension_wise_distribution, correlation_preservation
        static = pd.read_parquet("outputs/cohort/static_features.parquet")
        numeric_cols = static.select_dtypes(include=[np.number]).columns[:5]
        subset = static[numeric_cols].head(200)

        # Compare data with itself (perfect fidelity)
        result = dimension_wise_distribution(subset, subset)
        assert result["avg_pvalue"] > 0.5

        corr_result = correlation_preservation(subset, subset)
        assert corr_result["frobenius_distance"] < 0.01

    def test_clinical_rules(self):
        from src.evaluation.clinical_rules import check_clinical_rules
        static = pd.read_parquet("outputs/cohort/static_features.parquet")
        result = check_clinical_rules(static)
        # Real data should have very few violations
        assert result["total_violation_rate"] < 0.05

    def test_utility_metrics(self):
        from src.evaluation.utility import tstr_evaluation
        train = pd.read_parquet("outputs/cohort/static_features_train.parquet")
        test = pd.read_parquet("outputs/cohort/static_features_test.parquet")
        # Use train as both "real" and "synthetic" — should get near-perfect scores
        result = tstr_evaluation(train.head(500), train.head(500), test.head(200),
                                target="hospital_expire_flag")
        assert result["tstr_auc"] > 0.4

    def test_privacy_metrics(self):
        from src.evaluation.privacy import nearest_neighbor_distance
        static = pd.read_parquet("outputs/cohort/static_features.parquet")
        sub1 = static.head(100)
        sub2 = static.tail(100)
        result = nearest_neighbor_distance(sub1, sub2)
        assert result["mean_dcr"] > 0


class TestOutputFiles:
    """Verify all expected output files exist."""

    def test_cohort_outputs(self):
        outputs = Path("outputs/cohort")
        assert (outputs / "stroke_cohort.parquet").exists()
        assert (outputs / "static_features.parquet").exists()
        assert (outputs / "timeseries.parquet").exists()
        assert (outputs / "timeseries_processed.parquet").exists()
        assert (outputs / "static_features_train.parquet").exists()
        assert (outputs / "static_features_val.parquet").exists()
        assert (outputs / "static_features_test.parquet").exists()
        assert (outputs / "norm_params.json").exists()
