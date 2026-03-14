#!/usr/bin/env python3
"""Full evaluation pipeline for the stroke digital twin manuscript.

Fits BN, CTGAN, TVAE on training data, generates M=10 synthetic datasets
per model, computes all metrics per dataset, and pools results using
Rubin's combining rules (Reiter 2003 variant for synthetic data).
Results are saved to JSON and CSV tables with pooled estimates and 95%
confidence intervals.
"""

import sys
import os
import json
import time
import warnings
import traceback

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.models.bayesian_net import StrokeProfileBN
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
from src.models.dgan_model import StrokeTimeSeriesDGAN
from src.evaluation.fidelity import (
    dimension_wise_distribution,
    correlation_preservation,
    discriminator_score,
    medical_concept_abundance,
)
from src.evaluation.clinical_rules import (
    check_clinical_rules,
    inverse_normalize,
    load_norm_params,
)
from src.evaluation.utility import tstr_evaluation
from src.evaluation.privacy import (
    membership_inference_attack,
    nearest_neighbor_distance,
    attribute_inference_attack,
)
from src.evaluation.temporal import dtw_distance_matrix, autocorrelation_comparison
from src.evaluation.rubins_rules import pool_metric_dict


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COHORT_DIR = os.path.join(PROJECT_ROOT, "outputs", "cohort")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")
os.makedirs(TABLES_DIR, exist_ok=True)

ID_COLS = ["subject_id", "hadm_id", "stay_id"]

# Number of independently generated synthetic datasets per model.
# Following El Emam et al. (2024) recommendation of "at least 10".
M_DATASETS = 10

# Columns that are one-hot encoded stroke_subtype indicators
STROKE_OHE = [
    "stroke_subtype_ich",
    "stroke_subtype_ischemic",
    "stroke_subtype_other",
    "stroke_subtype_sah",
    "stroke_subtype_tia",
]

GENDER_OHE = ["gender_F", "gender_M"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reverse_ohe_stroke(df: pd.DataFrame) -> pd.Series:
    """Reverse one-hot encoding of stroke_subtype columns."""
    ohe_cols = [c for c in STROKE_OHE if c in df.columns]
    if not ohe_cols:
        return pd.Series("unknown", index=df.index)
    return df[ohe_cols].idxmax(axis=1).str.replace("stroke_subtype_", "")


def _reverse_ohe_gender(df: pd.DataFrame) -> pd.Series:
    """Reverse one-hot encoding of gender columns."""
    ohe_cols = [c for c in GENDER_OHE if c in df.columns]
    if not ohe_cols:
        return pd.Series("M", index=df.index)
    return df[ohe_cols].idxmax(axis=1).str.replace("gender_", "")


def _add_eval_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Add reverse-OHE stroke_subtype and gender columns for evaluation."""
    result = df.copy()
    result["stroke_subtype"] = _reverse_ohe_stroke(df)
    result["gender"] = _reverse_ohe_gender(df)
    return result


def _prepare_bn_data(
    df_ohe: pd.DataFrame,
    df_full: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare data for BN by merging original categoricals from full dataset."""
    full_ids = set(df_full["stay_id"])
    ohe_ids = set(df_ohe["stay_id"])
    common_ids = full_ids & ohe_ids

    bn_feats = StrokeProfileBN.BN_FEATURES
    available = [c for c in bn_feats if c in df_full.columns]
    subset = df_full[df_full["stay_id"].isin(common_ids)][["stay_id"] + available].copy()
    subset = subset.drop_duplicates(subset="stay_id")
    return subset


def _get_numeric_cols(df: pd.DataFrame) -> list:
    """Get numeric columns excluding IDs."""
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in ID_COLS]


def _coerce_bn_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert category dtypes in BN output to numeric/string."""
    result = df.copy()
    for col in result.columns:
        if result[col].dtype.name == "category":
            try:
                result[col] = pd.to_numeric(result[col])
            except (ValueError, TypeError):
                result[col] = result[col].astype(str)
    return result


def _safe_json(obj):
    """Make object JSON serializable."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict()
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    return obj


def _clean_for_json(d: dict) -> dict:
    """Recursively clean dict for JSON serialization."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _clean_for_json(v)
        elif isinstance(v, (list, tuple)):
            out[k] = [_safe_json(x) for x in v]
        else:
            out[k] = _safe_json(v)
    return out


def _pooled_val(pooled: dict, key: str) -> str:
    """Format a pooled metric as 'estimate [CI_lo, CI_hi]'."""
    entry = pooled.get(key, {})
    est = entry.get("pooled_estimate")
    if est is None:
        return ""
    lo = entry.get("ci_lower")
    hi = entry.get("ci_upper")
    if lo is not None and hi is not None:
        return f"{est:.4f} [{lo:.4f}, {hi:.4f}]"
    return f"{est:.4f}"


# ---------------------------------------------------------------------------
# Per-dataset metric computation helpers
# ---------------------------------------------------------------------------


def _compute_fidelity_single(
    model_name: str,
    synth_df: pd.DataFrame,
    real_test_numeric: pd.DataFrame,
    test_bn: pd.DataFrame,
    test_ohe_eval: pd.DataFrame,
) -> dict:
    """Compute fidelity metrics for a single synthetic dataset."""
    metrics: dict = {}

    if model_name == "BN":
        bn_num_cols = [
            c
            for c in synth_df.columns
            if synth_df[c].dtype in [np.float64, np.int64, np.float32, np.int32]
            and c not in ID_COLS
        ]
        real_for_bn = test_bn[[c for c in bn_num_cols if c in test_bn.columns]]
        synth_for_bn = synth_df[[c for c in bn_num_cols if c in synth_df.columns]]

        dwd = dimension_wise_distribution(real_for_bn, synth_for_bn)
        cp = correlation_preservation(real_for_bn, synth_for_bn)
        ds = discriminator_score(real_for_bn, synth_for_bn)

        if "stroke_subtype" in synth_df.columns and "stroke_subtype" in test_bn.columns:
            mca = medical_concept_abundance(
                test_bn,
                synth_df,
                "stroke_subtype",
            )
            metrics["mca_manhattan"] = mca["manhattan_distance"]
        else:
            metrics["mca_manhattan"] = 0.0
    else:
        synth_eval = _add_eval_cols(synth_df)
        synth_num = synth_df[_get_numeric_cols(synth_df)]
        dwd = dimension_wise_distribution(real_test_numeric, synth_num)
        cp = correlation_preservation(real_test_numeric, synth_num)
        ds = discriminator_score(real_test_numeric, synth_num)

        mca = medical_concept_abundance(
            test_ohe_eval,
            synth_eval,
            "stroke_subtype",
        )
        metrics["mca_manhattan"] = mca["manhattan_distance"]

    metrics["avg_ks_pvalue"] = dwd["avg_pvalue"]
    metrics["frobenius_distance"] = cp["frobenius_distance"]
    metrics["discriminator_auc"] = ds["auc"]
    return metrics


def _compute_plausibility_single(
    model_name: str,
    synth_df: pd.DataFrame,
    norm_params: dict,
) -> dict:
    """Compute plausibility metrics for a single synthetic dataset."""
    use_norm = norm_params if model_name != "BN" else None
    df_for_check = synth_df if model_name == "BN" else _add_eval_cols(synth_df)
    cr = check_clinical_rules(df_for_check, norm_params=use_norm)
    return {"total_violation_rate": cr["total_violation_rate"]}


def _compute_utility_single(
    model_name: str,
    synth_df: pd.DataFrame,
    train_ohe: pd.DataFrame,
    test_ohe: pd.DataFrame,
) -> dict:
    """Compute utility (TSTR) metrics for a single synthetic dataset."""
    train_cols = [c for c in train_ohe.columns if c not in ID_COLS]

    if model_name == "BN":
        bn_num = [
            c
            for c in synth_df.columns
            if c not in ID_COLS
            and synth_df[c].dtype in [np.float64, np.int64, np.float32, np.int32]
            and c != "hospital_expire_flag"
        ]
        common_feat = [c for c in bn_num if c in train_ohe.columns and c in test_ohe.columns]
        if "hospital_expire_flag" in synth_df.columns and len(common_feat) > 0:
            target_col = "hospital_expire_flag"
            bn_synth = synth_df[common_feat + [target_col]].copy()
            bn_train = train_ohe[common_feat + [target_col]].copy()
            bn_test = test_ohe[common_feat + [target_col]].copy()
            tstr = tstr_evaluation(bn_train, bn_synth, bn_test)
        else:
            tstr = {"trtr_auc": 0, "tstr_auc": 0, "auc_gap": 0}
    else:
        if "hospital_expire_flag" not in synth_df.columns:
            tstr = {"trtr_auc": 0, "tstr_auc": 0, "auc_gap": 0}
        else:
            try:
                tstr = tstr_evaluation(
                    train_ohe[train_cols],
                    synth_df,
                    test_ohe[train_cols],
                )
            except ValueError:
                tstr = {"trtr_auc": 0, "tstr_auc": 0, "auc_gap": 0}

    return {
        "trtr_auc": tstr.get("trtr_auc", 0),
        "tstr_auc": tstr.get("tstr_auc", 0),
        "auc_gap": tstr.get("auc_gap", 0),
    }


def _compute_privacy_single(
    model_name: str,
    synth_df: pd.DataFrame,
    train_ohe_clinical: pd.DataFrame,
    norm_params: dict,
) -> dict:
    """Compute privacy metrics for a single synthetic dataset."""
    if model_name == "BN":
        bn_num = [
            c
            for c in synth_df.columns
            if synth_df[c].dtype in [np.float64, np.int64, np.float32, np.int32]
            and c not in ID_COLS
        ]
        common = [c for c in bn_num if c in train_ohe_clinical.columns]
        if common:
            real_priv = train_ohe_clinical[common]
            synth_priv = synth_df[common]
            mia = membership_inference_attack(real_priv, synth_priv)
            nnd = nearest_neighbor_distance(real_priv, synth_priv)

            qi = [c for c in common if c != "hospital_expire_flag"][:8]
            if "hospital_expire_flag" in synth_df.columns:
                aia = attribute_inference_attack(
                    train_ohe_clinical,
                    synth_df,
                    "hospital_expire_flag",
                    qi,
                )
            else:
                aia = {"aia_accuracy": 0}
        else:
            mia = {"mia_f1": 0}
            nnd = {
                "mean_dcr": 0,
                "median_dcr": 0,
                "min_dcr": 0,
                "p5_dcr": 0,
            }
            aia = {"aia_accuracy": 0}
    else:
        synth_clinical = inverse_normalize(synth_df, norm_params)
        real_num = train_ohe_clinical[_get_numeric_cols(train_ohe_clinical)]
        synth_num = synth_clinical[_get_numeric_cols(synth_clinical)]
        mia = membership_inference_attack(real_num, synth_num)
        nnd = nearest_neighbor_distance(real_num, synth_num)
        qi = [c for c in _get_numeric_cols(train_ohe_clinical) if c != "hospital_expire_flag"][:10]
        aia = attribute_inference_attack(
            train_ohe_clinical,
            synth_clinical,
            "hospital_expire_flag",
            qi,
        )

    return {
        "mia_f1": mia["mia_f1"],
        "mean_dcr": nnd["mean_dcr"],
        "median_dcr": nnd["median_dcr"],
        "min_dcr": nnd["min_dcr"],
        "p5_dcr": nnd["p5_dcr"],
        "aia_accuracy": aia["aia_accuracy"],
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():  # noqa: C901
    results = {}
    t0 = time.time()

    # ==================================================================
    # 1. Load data
    # ==================================================================
    print("=" * 70)
    print("STEP 1: Loading data")
    print("=" * 70)

    train_ohe = pd.read_parquet(
        os.path.join(COHORT_DIR, "static_features_train.parquet"),
    )
    test_ohe = pd.read_parquet(
        os.path.join(COHORT_DIR, "static_features_test.parquet"),
    )
    full_static = pd.read_parquet(
        os.path.join(COHORT_DIR, "static_features.parquet"),
    )
    ts_df = pd.read_parquet(
        os.path.join(COHORT_DIR, "timeseries_processed.parquet"),
    )

    print(f"  Train (OHE):  {train_ohe.shape}")
    print(f"  Test  (OHE):  {test_ohe.shape}")
    print(f"  Full static:  {full_static.shape}")
    print(f"  Time-series:  {ts_df.shape}")

    n_test = len(test_ohe)

    # Load normalization parameters for inverse-normalization.
    # Required for correct plausibility and privacy evaluation.
    norm_params_path = os.path.join(COHORT_DIR, "norm_params.json")
    if not os.path.exists(norm_params_path):
        raise FileNotFoundError(
            f"{norm_params_path} not found. This file is required for "
            "correct evaluation of clinical plausibility and privacy."
        )
    norm_params = load_norm_params(norm_params_path)
    print(f"  Loaded norm_params: {len(norm_params)} columns")

    # Prepare BN-compatible data (original categoricals)
    train_bn = _prepare_bn_data(train_ohe, full_static)
    test_bn = _prepare_bn_data(test_ohe, full_static)
    print(f"  BN train: {train_bn.shape}, BN test: {test_bn.shape}")

    # Add stroke_subtype and gender columns for evaluation
    test_ohe_eval = _add_eval_cols(test_ohe)

    # ==================================================================
    # 2. Fit models (once each) and generate M synthetic datasets
    # ==================================================================
    print("\n" + "=" * 70)
    print(f"STEP 2: Fitting models and generating {M_DATASETS} synthetic datasets each")
    print("=" * 70)

    # --- BN ---
    print("\n  [BN] Fitting Bayesian Network...")
    t1 = time.time()
    bn = StrokeProfileBN(max_indegree=3)
    bn.fit(train_bn)
    bn_time = time.time() - t1
    print(f"  [BN] Fitted in {bn_time:.1f}s. DAG edges: {len(bn.get_dag())}")

    print(f"  [BN] Generating {M_DATASETS} synthetic datasets...")
    synth_bn_datasets = []
    for i in range(M_DATASETS):
        raw = bn.sample(n_test, seed=42 + i)
        synth_bn_datasets.append(_coerce_bn_dtypes(raw))
    print(f"  [BN] Done. Each dataset shape: {synth_bn_datasets[0].shape}")

    # --- CTGAN ---
    print("\n  [CTGAN] Fitting CTGAN (epochs=50)...")
    t1 = time.time()
    train_ctgan_cols = [c for c in train_ohe.columns if c not in ID_COLS]
    train_ctgan_df = train_ohe[train_ctgan_cols].copy()
    ctgan_meta = SingleTableMetadata()
    ctgan_meta.detect_from_dataframe(train_ctgan_df)
    ctgan_model = CTGANSynthesizer(
        ctgan_meta,
        epochs=50,
        batch_size=500,
        pac=1,
        verbose=False,
    )
    ctgan_model.fit(train_ctgan_df)
    ctgan_time = time.time() - t1
    print(f"  [CTGAN] Fitted in {ctgan_time:.1f}s.")

    print(f"  [CTGAN] Generating {M_DATASETS} synthetic datasets...")
    synth_ctgan_datasets = [ctgan_model.sample(num_rows=n_test) for _ in range(M_DATASETS)]
    print(f"  [CTGAN] Done. Each dataset shape: {synth_ctgan_datasets[0].shape}")

    # --- TVAE ---
    print("\n  [TVAE] Fitting TVAE (epochs=50)...")
    t1 = time.time()
    tvae_meta = SingleTableMetadata()
    tvae_meta.detect_from_dataframe(train_ctgan_df)
    tvae_model = TVAESynthesizer(tvae_meta, epochs=50, batch_size=500)
    tvae_model.fit(train_ctgan_df)
    tvae_time = time.time() - t1
    print(f"  [TVAE] Fitted in {tvae_time:.1f}s.")

    print(f"  [TVAE] Generating {M_DATASETS} synthetic datasets...")
    synth_tvae_datasets = [tvae_model.sample(num_rows=n_test) for _ in range(M_DATASETS)]
    print(f"  [TVAE] Done. Each dataset shape: {synth_tvae_datasets[0].shape}")

    results["model_training_times"] = {
        "bn_seconds": round(bn_time, 1),
        "ctgan_seconds": round(ctgan_time, 1),
        "tvae_seconds": round(tvae_time, 1),
    }
    results["m_datasets"] = M_DATASETS

    datasets_map = {
        "BN": synth_bn_datasets,
        "CTGAN": synth_ctgan_datasets,
        "TVAE": synth_tvae_datasets,
    }

    # ==================================================================
    # 2b. BN edge analysis for paradoxical associations
    # ==================================================================
    print("\n" + "=" * 70)
    print("STEP 2b: BN edge analysis (paradoxical associations)")
    print("=" * 70)

    edge_analysis = bn.analyse_edges(
        target="hospital_expire_flag",
        parents_of_interest=[
            "has_hypertension",
            "has_dyslipidemia",
            "has_diabetes",
            "has_afib",
        ],
    )
    results["bn_edge_analysis"] = edge_analysis
    print(f"  Edges into hospital_expire_flag: {edge_analysis['edges']}")
    print(f"  Parents analysed: {edge_analysis['parents']}")
    cpd_summary = edge_analysis.get("cpd_summary", {})
    if "conditional_probabilities" in cpd_summary:
        for combo, probs in cpd_summary["conditional_probabilities"].items():
            print(f"    {combo}: {probs}")

    # ==================================================================
    # 3. FIDELITY METRICS (pooled across M datasets)
    # ==================================================================
    print("\n" + "=" * 70)
    print(f"STEP 3: Computing fidelity metrics (pooled over {M_DATASETS} datasets)")
    print("=" * 70)

    real_test_numeric = test_ohe[_get_numeric_cols(test_ohe)]

    fidelity_pooled: dict = {}
    fidelity_per_dataset: dict = {}
    for model_name in ["BN", "CTGAN", "TVAE"]:
        print(f"\n  [{model_name}] Computing fidelity across {M_DATASETS} datasets...")
        per_ds = []
        for synth_df in datasets_map[model_name]:
            m = _compute_fidelity_single(
                model_name,
                synth_df,
                real_test_numeric,
                test_bn,
                test_ohe_eval,
            )
            per_ds.append(m)

        fidelity_per_dataset[model_name] = per_ds
        pooled = pool_metric_dict(per_ds)
        fidelity_pooled[model_name] = pooled

        for key in [
            "avg_ks_pvalue",
            "frobenius_distance",
            "discriminator_auc",
            "mca_manhattan",
        ]:
            print(f"    {key}: {_pooled_val(pooled, key)}")

    results["fidelity"] = fidelity_pooled

    # ==================================================================
    # 4. CLINICAL PLAUSIBILITY (pooled across M datasets)
    # ==================================================================
    print("\n" + "=" * 70)
    print(f"STEP 4: Clinical plausibility rules (pooled over {M_DATASETS} datasets)")
    print("=" * 70)

    # Real data: computed once (fixed dataset, no pooling needed)
    cr_real = check_clinical_rules(test_ohe_eval, norm_params=norm_params)
    clinical_real = {
        "total_violation_rate": round(cr_real["total_violation_rate"], 4),
        "total_violations": cr_real["total_violations"],
        "per_rule": {
            rule: {
                "violations": info["violations"],
                "violation_rate": round(info["violation_rate"], 4),
            }
            for rule, info in cr_real["per_rule"].items()
        },
    }
    print(
        f"  [Real] Total violation rate: "
        f"{cr_real['total_violation_rate']:.4f} "
        f"({cr_real['total_violations']} violations)"
    )

    clinical_pooled: dict = {}
    clinical_per_dataset: dict = {}
    for model_name in ["BN", "CTGAN", "TVAE"]:
        print(f"  [{model_name}] Computing plausibility across {M_DATASETS} datasets...")
        per_ds = []
        for synth_df in datasets_map[model_name]:
            m = _compute_plausibility_single(
                model_name,
                synth_df,
                norm_params,
            )
            per_ds.append(m)

        clinical_per_dataset[model_name] = per_ds
        pooled = pool_metric_dict(per_ds)
        clinical_pooled[model_name] = pooled
        print(f"    total_violation_rate: {_pooled_val(pooled, 'total_violation_rate')}")

    results["clinical_plausibility"] = {
        "Real": clinical_real,
        "pooled": clinical_pooled,
    }

    # ==================================================================
    # 5. UTILITY / TSTR (pooled across M datasets)
    # ==================================================================
    print("\n" + "=" * 70)
    print(f"STEP 5: Utility (TSTR) evaluation (pooled over {M_DATASETS} datasets)")
    print("=" * 70)

    utility_pooled: dict = {}
    utility_per_dataset: dict = {}
    for model_name in ["BN", "CTGAN", "TVAE"]:
        print(f"\n  [{model_name}] Running TSTR across {M_DATASETS} datasets...")
        per_ds = []
        for synth_df in datasets_map[model_name]:
            m = _compute_utility_single(
                model_name,
                synth_df,
                train_ohe,
                test_ohe,
            )
            per_ds.append(m)

        utility_per_dataset[model_name] = per_ds
        pooled = pool_metric_dict(per_ds)
        utility_pooled[model_name] = pooled

        for key in ["trtr_auc", "tstr_auc", "auc_gap"]:
            print(f"    {key}: {_pooled_val(pooled, key)}")

    results["utility"] = utility_pooled

    # ==================================================================
    # 6. PRIVACY METRICS (pooled across M datasets)
    # ==================================================================
    print("\n" + "=" * 70)
    print(f"STEP 6: Privacy metrics (pooled over {M_DATASETS} datasets)")
    print("=" * 70)

    # Inverse-normalize training data once
    train_ohe_clinical = inverse_normalize(train_ohe, norm_params)

    privacy_pooled: dict = {}
    privacy_per_dataset: dict = {}
    for model_name in ["BN", "CTGAN", "TVAE"]:
        print(f"\n  [{model_name}] Computing privacy across {M_DATASETS} datasets...")
        per_ds = []
        for synth_df in datasets_map[model_name]:
            m = _compute_privacy_single(
                model_name,
                synth_df,
                train_ohe_clinical,
                norm_params,
            )
            per_ds.append(m)

        privacy_per_dataset[model_name] = per_ds
        pooled = pool_metric_dict(per_ds)
        privacy_pooled[model_name] = pooled

        for key in ["mia_f1", "mean_dcr", "aia_accuracy"]:
            print(f"    {key}: {_pooled_val(pooled, key)}")

    results["privacy"] = privacy_pooled

    # ==================================================================
    # 7. TEMPORAL METRICS (DGAN -- single evaluation, not pooled)
    # ==================================================================
    print("\n" + "=" * 70)
    print("STEP 7: Temporal evaluation (DGAN)")
    print("=" * 70)

    temporal = {}
    try:
        ts_features = ["hr", "sbp", "dbp", "map", "rr", "spo2", "temp_c"]
        meta_features = ["anchor_age", "hospital_expire_flag"]

        stay_counts = ts_df.groupby("stay_id").size()
        valid_stays = stay_counts[stay_counts >= 24].index.tolist()
        print(f"  Stays with >=24 timesteps: {len(valid_stays)}")

        np.random.seed(42)
        if len(valid_stays) > 200:
            selected_stays = np.random.choice(
                valid_stays,
                200,
                replace=False,
            )
        else:
            selected_stays = valid_stays[:200]
        print(f"  Using {len(selected_stays)} patients for DGAN")

        seq_len = 24
        sequences = []
        metadata_list = []
        full_static_map = full_static.set_index("stay_id")

        for sid in selected_stays:
            ts_patient = ts_df[ts_df["stay_id"] == sid].sort_values("hour")
            ts_vals = ts_patient[ts_features].fillna(method="ffill").fillna(0).values
            if len(ts_vals) < seq_len:
                continue
            ts_vals = ts_vals[:seq_len]
            sequences.append(ts_vals)

            if sid in full_static_map.index:
                row = full_static_map.loc[sid]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                meta = [
                    float(row.get("anchor_age", 65)),
                    float(row.get("hospital_expire_flag", 0)),
                ]
            else:
                meta = [65.0, 0.0]
            metadata_list.append(meta)

        sequences = np.array(sequences, dtype=np.float32)
        metadata_arr = np.array(metadata_list, dtype=np.float32)
        print(f"  Sequences shape: {sequences.shape}")
        print(f"  Metadata shape:  {metadata_arr.shape}")

        seq_min = sequences.min(axis=(0, 1), keepdims=True)
        seq_max = sequences.max(axis=(0, 1), keepdims=True)
        seq_range = seq_max - seq_min
        seq_range[seq_range == 0] = 1
        sequences_norm = 2 * (sequences - seq_min) / seq_range - 1

        meta_min = metadata_arr.min(axis=0, keepdims=True)
        meta_max = metadata_arr.max(axis=0, keepdims=True)
        meta_range = meta_max - meta_min
        meta_range[meta_range == 0] = 1
        metadata_norm = 2 * (metadata_arr - meta_min) / meta_range - 1

        print("  Training DGAN (epochs=10, hidden_dim=32)...")
        t1 = time.time()
        dgan = StrokeTimeSeriesDGAN(
            n_features=len(ts_features),
            n_metadata=len(meta_features),
            seq_len=seq_len,
            noise_dim=64,
            hidden_dim=32,
            epochs=10,
            batch_size=min(32, len(sequences_norm)),
            lr=0.0002,
        )
        dgan.train(metadata_norm, sequences_norm)
        dgan_time = time.time() - t1
        print(f"  DGAN trained in {dgan_time:.1f}s")

        synth_ts = dgan.generate(metadata_norm)
        print(f"  Synthetic TS shape: {synth_ts.shape}")

        print("  Computing DTW distances...")
        dtw_res = dtw_distance_matrix(
            sequences_norm,
            synth_ts,
            n_samples=min(100, len(sequences_norm)),
        )
        temporal["dtw"] = {k: round(v, 4) for k, v in dtw_res.items()}
        print(f"    Mean DTW: {dtw_res['mean_dtw']:.4f}")

        print("  Computing autocorrelation comparison...")
        acf_res = autocorrelation_comparison(
            sequences_norm,
            synth_ts,
            ts_features,
            max_lag=12,
        )
        temporal["autocorrelation"] = {}
        for feat, vals in acf_res.items():
            temporal["autocorrelation"][feat] = {
                "mean_diff": round(vals["mean_diff"], 4),
                "max_diff": round(vals["max_diff"], 4),
            }
            print(f"    {feat}: mean_diff={vals['mean_diff']:.4f}, max_diff={vals['max_diff']:.4f}")

        temporal["dgan_training_time"] = round(dgan_time, 1)
        temporal["n_patients"] = len(sequences_norm)
        temporal["seq_len"] = seq_len
        results["model_training_times"]["dgan_seconds"] = round(
            dgan_time,
            1,
        )

    except Exception as e:
        print(f"  ERROR in temporal evaluation: {e}")
        traceback.print_exc()
        temporal["error"] = str(e)

    results["temporal"] = temporal

    # ==================================================================
    # 8. Save JSON results
    # ==================================================================
    print("\n" + "=" * 70)
    print("STEP 8: Saving results")
    print("=" * 70)

    results["per_dataset"] = {
        "fidelity": fidelity_per_dataset,
        "clinical_plausibility": clinical_per_dataset,
        "utility": utility_per_dataset,
        "privacy": privacy_per_dataset,
    }

    results_clean = _clean_for_json(results)
    json_path = os.path.join(OUTPUT_DIR, "evaluation_results.json")
    with open(json_path, "w") as f:
        json.dump(results_clean, f, indent=2, default=str)
    print(f"  Saved: {json_path}")

    # ==================================================================
    # 9. Generate CSV tables (with pooled estimates and 95% CIs)
    # ==================================================================

    # --- Table 4: Fidelity comparison (pooled) ---
    table4_rows = []
    for model in ["BN", "CTGAN", "TVAE"]:
        p = fidelity_pooled[model]
        table4_rows.append(
            {
                "Model": model,
                "Avg KS p-value": _pooled_val(p, "avg_ks_pvalue"),
                "Frobenius Distance": _pooled_val(
                    p,
                    "frobenius_distance",
                ),
                "Discriminator AUC": _pooled_val(
                    p,
                    "discriminator_auc",
                ),
                "MCA Manhattan": _pooled_val(p, "mca_manhattan"),
            }
        )
    table4 = pd.DataFrame(table4_rows)
    table4_path = os.path.join(
        TABLES_DIR,
        "table4_fidelity_comparison.csv",
    )
    table4.to_csv(table4_path, index=False)
    print(f"  Saved: {table4_path}")

    # --- Table 5: Clinical rules (Real + pooled synthetic) ---
    table5_rows = [
        {
            "Dataset": "Real",
            "Total Violation Rate": (f"{clinical_real['total_violation_rate']:.4f}"),
        }
    ]
    for model in ["BN", "CTGAN", "TVAE"]:
        p = clinical_pooled[model]
        table5_rows.append(
            {
                "Dataset": model,
                "Total Violation Rate": _pooled_val(
                    p,
                    "total_violation_rate",
                ),
            }
        )
    table5 = pd.DataFrame(table5_rows)
    table5_path = os.path.join(TABLES_DIR, "table5_clinical_rules.csv")
    table5.to_csv(table5_path, index=False)
    print(f"  Saved: {table5_path}")

    # --- Table 6: TSTR results (pooled) ---
    table6_rows = []
    for model in ["BN", "CTGAN", "TVAE"]:
        p = utility_pooled[model]
        table6_rows.append(
            {
                "Model": model,
                "Mean TRTR AUC": _pooled_val(p, "trtr_auc"),
                "Mean TSTR AUC": _pooled_val(p, "tstr_auc"),
                "AUC Gap": _pooled_val(p, "auc_gap"),
            }
        )
    table6 = pd.DataFrame(table6_rows)
    table6_path = os.path.join(TABLES_DIR, "table6_tstr_results.csv")
    table6.to_csv(table6_path, index=False)
    print(f"  Saved: {table6_path}")

    # --- Table 7: Privacy metrics (pooled) ---
    table7_rows = []
    for model in ["BN", "CTGAN", "TVAE"]:
        p = privacy_pooled[model]
        table7_rows.append(
            {
                "Model": model,
                "MIA F1": _pooled_val(p, "mia_f1"),
                "Mean DCR": _pooled_val(p, "mean_dcr"),
                "Median DCR": _pooled_val(p, "median_dcr"),
                "5th %ile DCR": _pooled_val(p, "p5_dcr"),
                "AIA Accuracy": _pooled_val(p, "aia_accuracy"),
            }
        )
    table7 = pd.DataFrame(table7_rows)
    table7_path = os.path.join(TABLES_DIR, "table7_privacy_metrics.csv")
    table7.to_csv(table7_path, index=False)
    print(f"  Saved: {table7_path}")

    total_time = time.time() - t0

    # ==================================================================
    # 10. Print summary
    # ==================================================================
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    print(f"\nTotal runtime: {total_time:.0f}s ({total_time / 60:.1f} min)")
    print(f"Training samples: {len(train_ohe)}, Test samples: {len(test_ohe)}")
    print(f"Synthetic datasets per model (Rubin's rules): {M_DATASETS}")

    print("\n--- FIDELITY (Table 4) --- pooled estimates [95% CI] ---")
    print(table4.to_string(index=False))

    print("\n--- CLINICAL RULES (Table 5) --- pooled estimates [95% CI] ---")
    print(table5.to_string(index=False))

    print("\n--- UTILITY / TSTR (Table 6) --- pooled estimates [95% CI] ---")
    print(table6.to_string(index=False))

    print("\n--- PRIVACY (Table 7) --- pooled estimates [95% CI] ---")
    print(table7.to_string(index=False))

    if "dtw" in temporal:
        print("\n--- TEMPORAL ---")
        print(f"  DTW: mean={temporal['dtw']['mean_dtw']}, median={temporal['dtw']['median_dtw']}")
        if "autocorrelation" in temporal:
            for feat, v in temporal["autocorrelation"].items():
                print(f"  ACF {feat}: mean_diff={v['mean_diff']}, max_diff={v['max_diff']}")

    if edge_analysis["edges"]:
        print("\n--- BN EDGE ANALYSIS (paradoxical associations) ---")
        for edge in edge_analysis["edges"]:
            print(f"  {edge[0]} -> {edge[1]}")
        cpd_sum = edge_analysis.get("cpd_summary", {})
        if "conditional_probabilities" in cpd_sum:
            for combo, probs in cpd_sum["conditional_probabilities"].items():
                print(f"    {combo}: {probs}")

    print("\n" + "=" * 70)
    print("ALL DONE. Results saved to:")
    print(f"  {json_path}")
    print(f"  {table4_path}")
    print(f"  {table5_path}")
    print(f"  {table6_path}")
    print(f"  {table7_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
