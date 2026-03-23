"""Preprocessing pipeline for stroke digital twin data."""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical columns."""
    cat_cols = [
        "gender",
        "race",
        "insurance",
        "stroke_subtype",
        "first_careunit",
        "admission_type",
    ]
    # Only encode columns that exist
    cols_to_encode = [c for c in cat_cols if c in df.columns]
    return pd.get_dummies(df, columns=cols_to_encode, drop_first=False, dtype=int)


def impute_missing_static(
    df: pd.DataFrame,
    method: str = "median",
    fill_values: dict | None = None,
) -> pd.DataFrame:
    """Impute missing values in static features.

    Parameters
    ----------
    df : pd.DataFrame
        Static features with potential missing values.
    method : str
        Imputation strategy for numeric columns.  One of ``"median"``
        (default) or ``"mean"``.  Categorical columns always use mode
        imputation regardless of this setting. Ignored when
        *fill_values* is provided.
    fill_values : dict, optional
        Pre-computed fill values per column (from ``fit_imputation``).
        When provided, these values are used directly instead of
        computing new statistics from *df*. This prevents data leakage
        when applying train-set statistics to val/test splits.

    Returns
    -------
    pd.DataFrame
        DataFrame with missing values filled.
    """
    if method not in ("median", "mean"):
        raise ValueError(f"Unsupported imputation method: {method!r}")

    result = df.copy()

    # Lab columns that should get missingness flags
    lab_cols = [c for c in df.columns if c.startswith("lab_") or c.endswith("_admit")]

    for col in lab_cols:
        if col in result.columns and result[col].isna().any():
            result[f"{col}_missing"] = result[col].isna().astype(int)
            if fill_values and col in fill_values:
                result[col] = result[col].fillna(fill_values[col])
            else:
                fv = result[col].median() if method == "median" else result[col].mean()
                result[col] = result[col].fillna(fv)

    # Other numeric columns — impute without missingness flag
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if result[col].isna().any():
            if fill_values and col in fill_values:
                result[col] = result[col].fillna(fill_values[col])
            else:
                fv = result[col].median() if method == "median" else result[col].mean()
                result[col] = result[col].fillna(fv)

    # Categorical columns — mode impute
    cat_cols = result.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        if result[col].isna().any():
            mode_val = result[col].mode()
            if len(mode_val) > 0:
                result[col] = result[col].fillna(mode_val.iloc[0])

    return result


def fit_imputation(
    train_df: pd.DataFrame,
    method: str = "median",
) -> dict:
    """Compute fill values from the training set only.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training-split static features.
    method : str
        ``"median"`` (default) or ``"mean"``.

    Returns
    -------
    dict
        Column name -> fill value mapping.
    """
    fill_values: dict = {}
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if train_df[col].isna().any():
            fv = train_df[col].median() if method == "median" else train_df[col].mean()
            fill_values[col] = float(fv)
    # Also capture lab/admit columns that might have been encoded
    lab_cols = [c for c in train_df.columns if c.startswith("lab_") or c.endswith("_admit")]
    for col in lab_cols:
        if col in train_df.columns and col not in fill_values and train_df[col].isna().any():
            fv = train_df[col].median() if method == "median" else train_df[col].mean()
            fill_values[col] = float(fv)
    return fill_values


def apply_imputation(
    df: pd.DataFrame,
    fill_values: dict,
    method: str = "median",
) -> pd.DataFrame:
    """Apply pre-computed fill values (wrapper around ``impute_missing_static``).

    Parameters
    ----------
    df : pd.DataFrame
        Static features with potential missing values.
    fill_values : dict
        Pre-computed fill values (from ``fit_imputation``).
    method : str
        Passed through for categorical mode imputation.

    Returns
    -------
    pd.DataFrame
        Imputed DataFrame.
    """
    return impute_missing_static(df, method=method, fill_values=fill_values)


def normalize_numeric(df: pd.DataFrame, columns: list = None) -> tuple[pd.DataFrame, dict]:
    """Min-max normalize numeric columns to [-1, 1].

    Returns (normalized_df, params_dict) where params contains min/max per column.
    """
    result = df.copy()
    if columns is None:
        columns = result.select_dtypes(include=[np.number]).columns.tolist()

    params = {}
    for col in columns:
        if col in result.columns:
            col_min = result[col].min()
            col_max = result[col].max()
            if col_max > col_min:
                result[col] = 2 * (result[col] - col_min) / (col_max - col_min) - 1
            params[col] = {"min": col_min, "max": col_max}

    return result, params


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train/val/test, stratified by hospital_expire_flag."""
    stratify_col = "hospital_expire_flag" if "hospital_expire_flag" in df.columns else None

    # First split: train+val vs test
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df[stratify_col] if stratify_col else None,
    )

    # Second split: train vs val
    val_ratio = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=val_ratio,
        random_state=seed,
        stratify=train_val[stratify_col] if stratify_col else None,
    )

    return train, val, test


def preprocess_timeseries(ts: pd.DataFrame) -> pd.DataFrame:
    """Preprocess time-series: forward-fill within each stay, clip outliers."""
    result = ts.copy()
    vital_cols = [
        "hr",
        "sbp",
        "dbp",
        "map",
        "rr",
        "spo2",
        "temp_c",
        "gcs_eye",
        "gcs_verbal",
        "gcs_motor",
        "gcs_total",
    ]

    existing_cols = [c for c in vital_cols if c in result.columns]

    # Forward-fill within each stay
    result = result.sort_values(["stay_id", "hour"])
    result[existing_cols] = result.groupby("stay_id")[existing_cols].ffill()

    # Clip to physiological ranges
    clip_ranges = {
        "hr": (20, 300),
        "sbp": (40, 300),
        "dbp": (20, 200),
        "map": (20, 250),
        "rr": (4, 60),
        "spo2": (50, 100),
        "temp_c": (30, 45),
        "gcs_eye": (1, 4),
        "gcs_verbal": (1, 5),
        "gcs_motor": (1, 6),
        "gcs_total": (3, 15),
    }
    for col, (lo, hi) in clip_ranges.items():
        if col in result.columns:
            result[col] = result[col].clip(lo, hi)

    return result


def preprocess_pipeline(config: dict) -> dict:
    """Run full preprocessing pipeline.

    Returns dict with processed DataFrames and normalization params.
    """
    cohort_path = Path(config["data"]["cohort_path"])

    # Load data
    static = pd.read_parquet(cohort_path / "static_features.parquet")
    ts = pd.read_parquet(cohort_path / "timeseries.parquet")

    # Drop columns not useful for modeling
    drop_cols = [
        "icd_code",
        "icd_version",
        "seq_num",
        "icd_title",
        "intime",
        "outtime",
        "dod",
        "admittime",
        "dischtime",
        "deathtime",
        "admission_location",
        "discharge_location",
    ]
    static = static.drop(columns=[c for c in drop_cols if c in static.columns])

    # Split static data (before encoding/imputation to prevent leakage)
    train_raw, val_raw, test_raw = split_data(
        static,
        test_size=config["evaluation"]["tstr_test_size"],
        seed=config["models"]["random_seed"],
    )

    # Impute missing values (fit on train, apply to val/test — no leakage)
    fill_values = fit_imputation(train_raw)
    train_imputed = apply_imputation(train_raw, fill_values)
    val_imputed = apply_imputation(val_raw, fill_values)
    test_imputed = apply_imputation(test_raw, fill_values)

    # Encode categoricals
    train_encoded = encode_categoricals(train_imputed)
    val_encoded = encode_categoricals(val_imputed)
    test_encoded = encode_categoricals(test_imputed)

    # Align columns (one-hot may create different columns across splits)
    all_cols = sorted(
        set(train_encoded.columns) | set(val_encoded.columns) | set(test_encoded.columns)
    )
    for col in all_cols:
        for df in [train_encoded, val_encoded, test_encoded]:
            if col not in df.columns:
                df[col] = 0
    train_encoded = train_encoded[all_cols]
    val_encoded = val_encoded[all_cols]
    test_encoded = test_encoded[all_cols]

    # Normalize numeric columns (fit on train only)
    exclude_from_norm = {
        "subject_id",
        "hadm_id",
        "stay_id",
        "hospital_expire_flag",
    }
    numeric_cols = [
        c
        for c in train_encoded.select_dtypes(include=[np.number]).columns
        if not c.startswith("has_") and not c.endswith("_missing") and c not in exclude_from_norm
    ]
    train_norm, norm_params = normalize_numeric(train_encoded, numeric_cols)

    # Apply same normalization to val/test using train params
    val_norm = val_encoded.copy()
    test_norm = test_encoded.copy()
    for col, params in norm_params.items():
        if params["max"] > params["min"]:
            for df in [val_norm, test_norm]:
                if col in df.columns:
                    df[col] = 2 * (df[col] - params["min"]) / (params["max"] - params["min"]) - 1

    # Preprocess time-series
    ts_processed = preprocess_timeseries(ts)

    # Save outputs
    out_path = cohort_path
    train_norm.to_parquet(out_path / "static_features_train.parquet", index=False)
    val_norm.to_parquet(out_path / "static_features_val.parquet", index=False)
    test_norm.to_parquet(out_path / "static_features_test.parquet", index=False)
    ts_processed.to_parquet(out_path / "timeseries_processed.parquet", index=False)

    # Save normalization params
    norm_params_serializable = {
        k: {kk: float(vv) for kk, vv in v.items()} for k, v in norm_params.items()
    }
    with open(out_path / "norm_params.json", "w") as f:
        json.dump(norm_params_serializable, f, indent=2)

    print(f"Train: {len(train_norm)}, Val: {len(val_norm)}, Test: {len(test_norm)}")
    print(f"Time-series: {len(ts_processed)} rows")
    print(f"Columns after encoding: {len(train_norm.columns)}")

    return {
        "static_train": train_norm,
        "static_val": val_norm,
        "static_test": test_norm,
        "ts_processed": ts_processed,
        "norm_params": norm_params,
    }
