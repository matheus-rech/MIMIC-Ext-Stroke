import pytest
import pandas as pd
import numpy as np


def test_dimension_wise_distribution_identical():
    from src.evaluation.fidelity import dimension_wise_distribution
    df = pd.DataFrame({"a": np.random.randn(100), "b": np.random.randn(100)})
    result = dimension_wise_distribution(df, df)
    # Identical distributions should have high p-values
    assert result["avg_pvalue"] > 0.9


def test_dimension_wise_distribution_different():
    from src.evaluation.fidelity import dimension_wise_distribution
    real = pd.DataFrame({"a": np.random.randn(100)})
    synth = pd.DataFrame({"a": np.random.randn(100) + 5})  # shifted
    result = dimension_wise_distribution(real, synth)
    assert result["avg_pvalue"] < 0.05


def test_correlation_preservation():
    from src.evaluation.fidelity import correlation_preservation
    np.random.seed(42)
    x = np.random.randn(200)
    real = pd.DataFrame({"a": x, "b": x + np.random.randn(200) * 0.1})
    synth = pd.DataFrame({"a": x, "b": x + np.random.randn(200) * 0.1})
    result = correlation_preservation(real, synth)
    assert result["frobenius_distance"] < 0.5


def test_discriminator_score_identical():
    from src.evaluation.fidelity import discriminator_score
    np.random.seed(42)
    df = pd.DataFrame({"a": np.random.randn(200), "b": np.random.randn(200)})
    result = discriminator_score(df, df)
    # Should be near 0.5 (can't distinguish)
    assert 0.3 < result["auc"] < 0.7


def test_clinical_rules_valid_data():
    from src.evaluation.clinical_rules import check_clinical_rules
    df = pd.DataFrame({
        "anchor_age": [65, 70, 55],
        "gcs_total": [12, 8, 15],
        "los": [2, 5, 1],
        "hospital_expire_flag": [0, 1, 0],
    })
    result = check_clinical_rules(df)
    assert result["total_violations"] == 0


def test_clinical_rules_detects_violations():
    from src.evaluation.clinical_rules import check_clinical_rules
    df = pd.DataFrame({
        "anchor_age": [5, 200, 65],  # 5 and 200 are violations
        "gcs_total": [2, 8, 16],      # 2 and 16 are violations
        "los": [-1, 5, 1],            # -1 is violation
    })
    result = check_clinical_rules(df)
    assert result["total_violations"] > 0


def test_medical_concept_abundance():
    from src.evaluation.fidelity import medical_concept_abundance
    real = pd.DataFrame({"stroke_subtype": ["ischemic"] * 70 + ["ich"] * 20 + ["sah"] * 10})
    synth = pd.DataFrame({"stroke_subtype": ["ischemic"] * 65 + ["ich"] * 25 + ["sah"] * 10})
    result = medical_concept_abundance(real, synth, "stroke_subtype")
    assert result["manhattan_distance"] < 0.2
