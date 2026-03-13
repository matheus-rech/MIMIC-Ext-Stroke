"""Tests for Bayesian Network static profile generation."""
import pytest
import pandas as pd


@pytest.fixture
def static_data():
    return pd.read_parquet("outputs/cohort/static_features.parquet")


@pytest.fixture
def train_data(static_data):
    from src.data.preprocess import split_data

    train, _, _ = split_data(static_data, test_size=0.3, val_size=0.1, seed=42)
    return train


def test_bn_learns_structure(train_data):
    from src.models.bayesian_net import StrokeProfileBN

    bn = StrokeProfileBN()
    bn.fit(train_data)
    assert len(bn.model.edges()) > 0
    assert "has_afib" in [n for n in bn.model.nodes()]


def test_bn_generates_samples(train_data):
    from src.models.bayesian_net import StrokeProfileBN

    bn = StrokeProfileBN()
    bn.fit(train_data)
    synthetic = bn.sample(n=100, seed=42)
    assert len(synthetic) == 100
    # Should have the BN feature columns
    assert "stroke_subtype" in synthetic.columns
    assert "has_hypertension" in synthetic.columns


def test_bn_preserves_rare_subtypes(train_data):
    from src.models.bayesian_net import StrokeProfileBN

    bn = StrokeProfileBN()
    bn.fit(train_data)
    synthetic = bn.sample(n=len(train_data), seed=42)
    # SAH is rare (~5%); should be present in synthetic
    if "stroke_subtype" in synthetic.columns:
        real_sah = (train_data["stroke_subtype"] == "sah").mean()
        synth_sah = (synthetic["stroke_subtype"] == "sah").mean()
        assert synth_sah > 0, "SAH subtype completely lost"
        assert (
            abs(synth_sah - real_sah) < 0.10
        ), f"SAH drift: {real_sah:.3f} vs {synth_sah:.3f}"


def test_bn_get_dag(train_data):
    from src.models.bayesian_net import StrokeProfileBN

    bn = StrokeProfileBN()
    bn.fit(train_data)
    edges = bn.get_dag()
    assert isinstance(edges, list)
    assert len(edges) > 0
    # Each edge is a tuple of two node names
    assert all(isinstance(e, tuple) and len(e) == 2 for e in edges)
