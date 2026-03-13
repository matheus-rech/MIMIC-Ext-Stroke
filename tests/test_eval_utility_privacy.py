# tests/test_eval_utility_privacy.py
import pytest
import numpy as np
import pandas as pd


def test_tstr_returns_metrics():
    from src.evaluation.utility import tstr_evaluation
    np.random.seed(42)
    n = 200
    real_train = pd.DataFrame({"f1": np.random.randn(n), "f2": np.random.randn(n),
                               "hospital_expire_flag": np.random.binomial(1, 0.15, n)})
    synth = pd.DataFrame({"f1": np.random.randn(n), "f2": np.random.randn(n),
                          "hospital_expire_flag": np.random.binomial(1, 0.15, n)})
    real_test = pd.DataFrame({"f1": np.random.randn(50), "f2": np.random.randn(50),
                              "hospital_expire_flag": np.random.binomial(1, 0.15, 50)})

    result = tstr_evaluation(real_train, synth, real_test, target="hospital_expire_flag")
    assert "tstr_auc" in result
    assert "trtr_auc" in result
    assert 0 <= result["tstr_auc"] <= 1


def test_membership_inference():
    from src.evaluation.privacy import membership_inference_attack
    np.random.seed(42)
    real = pd.DataFrame({"a": np.random.randn(100), "b": np.random.randn(100)})
    synth = pd.DataFrame({"a": np.random.randn(100) + 5, "b": np.random.randn(100) + 5})
    result = membership_inference_attack(real, synth)
    assert "mia_f1" in result
    assert 0 <= result["mia_f1"] <= 1


def test_nearest_neighbor_distance():
    from src.evaluation.privacy import nearest_neighbor_distance
    np.random.seed(42)
    real = pd.DataFrame({"a": np.random.randn(50), "b": np.random.randn(50)})
    synth = pd.DataFrame({"a": np.random.randn(50) + 10, "b": np.random.randn(50) + 10})
    result = nearest_neighbor_distance(real, synth)
    assert "mean_dcr" in result
    assert result["mean_dcr"] > 0


def test_dtw_distance():
    from src.evaluation.temporal import dtw_distance_matrix
    np.random.seed(42)
    # 10 sequences, 24 timesteps, 3 features
    real = np.random.randn(10, 24, 3)
    synth = np.random.randn(10, 24, 3)
    result = dtw_distance_matrix(real, synth)
    assert "mean_dtw" in result
    assert result["mean_dtw"] > 0


def test_autocorrelation_comparison():
    from src.evaluation.temporal import autocorrelation_comparison
    np.random.seed(42)
    real = np.random.randn(50, 24, 2)
    synth = np.random.randn(50, 24, 2)
    result = autocorrelation_comparison(real, synth, feature_names=["hr", "sbp"])
    assert "hr" in result
    assert "mean_diff" in result["hr"]
