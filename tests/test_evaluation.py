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


def test_inverse_normalize_roundtrip():
    from src.evaluation.clinical_rules import inverse_normalize
    norm_params = {
        "anchor_age": {"min": 18.0, "max": 100.0},
        "los": {"min": 0.5, "max": 30.0},
    }
    # Normalized data in [-1, 1]
    df_norm = pd.DataFrame({
        "anchor_age": [-1.0, 0.0, 1.0],
        "los": [-1.0, 0.0, 1.0],
    })
    df_clinical = inverse_normalize(df_norm, norm_params)
    assert abs(df_clinical["anchor_age"].iloc[0] - 18.0) < 0.01
    assert abs(df_clinical["anchor_age"].iloc[1] - 59.0) < 0.01
    assert abs(df_clinical["anchor_age"].iloc[2] - 100.0) < 0.01


def test_clinical_rules_with_norm_params():
    from src.evaluation.clinical_rules import check_clinical_rules
    norm_params = {
        "anchor_age": {"min": 18.0, "max": 100.0},
    }
    # Normalized value 0.0 maps to midpoint (59) — should be valid
    df_norm = pd.DataFrame({"anchor_age": [0.0, 0.5, -0.5]})
    result = check_clinical_rules(df_norm, norm_params=norm_params)
    assert result["per_rule"]["age_valid"]["violations"] == 0


def test_rubins_rules_pool_estimates():
    from src.evaluation.rubins_rules import pool_estimates
    estimates = [5.0, 5.2, 4.8, 5.1, 5.05]
    result = pool_estimates(estimates)
    assert abs(result["pooled_estimate"] - 5.03) < 0.01
    assert result["between_variance"] > 0
    assert result["ci_lower"] < result["pooled_estimate"]
    assert result["ci_upper"] > result["pooled_estimate"]
    assert result["m"] == 5


def test_rubins_rules_pool_metric_dict():
    from src.evaluation.rubins_rules import pool_metric_dict
    dicts = [
        {"mean_dcr": 5.1, "mia_f1": 0.55},
        {"mean_dcr": 5.3, "mia_f1": 0.53},
        {"mean_dcr": 5.0, "mia_f1": 0.56},
    ]
    pooled = pool_metric_dict(dicts)
    assert "mean_dcr" in pooled
    assert abs(pooled["mean_dcr"]["pooled_estimate"] - 5.133) < 0.01
    assert pooled["mia_f1"]["ci_lower"] < pooled["mia_f1"]["ci_upper"]
