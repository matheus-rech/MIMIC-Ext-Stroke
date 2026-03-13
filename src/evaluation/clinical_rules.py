"""Stroke-specific clinical plausibility rules."""
import pandas as pd


RULES = {
    "age_valid": lambda df: (
        (df["anchor_age"] >= 18) & (df["anchor_age"] <= 120)
        if "anchor_age" in df.columns
        else pd.Series(True, index=df.index)
    ),
    "gcs_valid": lambda df: (
        (df["gcs_total"] >= 3) & (df["gcs_total"] <= 15)
        if "gcs_total" in df.columns
        else pd.Series(True, index=df.index)
    ),
    "los_positive": lambda df: (
        df["los"] > 0 if "los" in df.columns else pd.Series(True, index=df.index)
    ),
    "sbp_gt_dbp": lambda df: (
        df["sbp"] > df["dbp"]
        if "sbp" in df.columns and "dbp" in df.columns
        else pd.Series(True, index=df.index)
    ),
    "spo2_range": lambda df: (
        (df["spo2"] >= 50) & (df["spo2"] <= 100)
        if "spo2" in df.columns
        else pd.Series(True, index=df.index)
    ),
    "hr_range": lambda df: (
        (df["hr"] >= 20) & (df["hr"] <= 300)
        if "hr" in df.columns
        else pd.Series(True, index=df.index)
    ),
    "temp_range": lambda df: (
        (df["temp_c"] >= 30) & (df["temp_c"] <= 45)
        if "temp_c" in df.columns
        else pd.Series(True, index=df.index)
    ),
}


def check_clinical_rules(df: pd.DataFrame) -> dict:
    """Check synthetic data against clinical plausibility rules.

    Returns dict with per-rule violations and total count.
    """
    results = {}
    total_violations = 0

    for name, rule_fn in RULES.items():
        valid = rule_fn(df)
        n_violations = (~valid).sum()
        results[name] = {
            "violations": int(n_violations),
            "violation_rate": float(n_violations / len(df)) if len(df) > 0 else 0,
        }
        total_violations += n_violations

    return {
        "per_rule": results,
        "total_violations": total_violations,
        "total_violation_rate": (
            total_violations / (len(df) * len(RULES)) if len(df) > 0 else 0
        ),
    }
