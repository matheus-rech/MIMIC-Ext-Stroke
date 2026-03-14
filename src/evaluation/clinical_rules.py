"""Stroke-specific clinical plausibility rules."""

import json
from pathlib import Path

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


def inverse_normalize(
    df: pd.DataFrame,
    norm_params: dict,
) -> pd.DataFrame:
    """Convert normalized [-1, 1] data back to the original clinical scale.

    Parameters
    ----------
    df : pd.DataFrame
        Data in the normalized [-1, 1] space.
    norm_params : dict
        Mapping of ``{column: {"min": float, "max": float}}`` produced
        during the preprocessing step.

    Returns
    -------
    pd.DataFrame
        Data on the original clinical scale.
    """
    result = df.copy()
    for col, params in norm_params.items():
        if col not in result.columns:
            continue
        col_min = params["min"]
        col_max = params["max"]
        if col_max > col_min:
            result[col] = (result[col] + 1) / 2 * (col_max - col_min) + col_min
    return result


def load_norm_params(path: str | Path) -> dict:
    """Load normalization parameters from a JSON file."""
    with open(path) as f:
        return json.load(f)


def check_clinical_rules(
    df: pd.DataFrame,
    norm_params: dict | None = None,
) -> dict:
    """Check synthetic data against clinical plausibility rules.

    If *norm_params* is provided the data is inverse-normalised back to the
    original clinical scale before applying the rules.  This is essential
    when the input data has been min-max normalised to [-1, 1].

    Returns dict with per-rule violations and total count.
    """
    if norm_params is not None:
        df = inverse_normalize(df, norm_params)

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
        "total_violation_rate": (total_violations / (len(df) * len(RULES)) if len(df) > 0 else 0),
    }
