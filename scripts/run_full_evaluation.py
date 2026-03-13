#!/usr/bin/env python3
"""Full evaluation pipeline for the stroke digital twin manuscript.

Fits BN, CTGAN, TVAE on training data, generates synthetic data,
and computes all metrics (fidelity, clinical plausibility, utility,
privacy, temporal). Results are saved to JSON and CSV tables.
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
from src.evaluation.clinical_rules import check_clinical_rules
from src.evaluation.utility import tstr_evaluation
from src.evaluation.privacy import (
    membership_inference_attack,
    nearest_neighbor_distance,
    attribute_inference_attack,
)
from src.evaluation.temporal import dtw_distance_matrix, autocorrelation_comparison


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
COHORT_DIR = os.path.join(PROJECT_ROOT, "outputs", "cohort")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")
os.makedirs(TABLES_DIR, exist_ok=True)

ID_COLS = ["subject_id", "hadm_id", "stay_id"]

# Columns that are one-hot encoded stroke_subtype indicators
STROKE_OHE = [
    "stroke_subtype_ich",
    "stroke_subtype_ischemic",
    "stroke_subtype_other",
    "stroke_subtype_sah",
    "stroke_subtype_tia",
]

GENDER_OHE = ["gender_F", "gender_M"]


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


def _prepare_bn_data(df_ohe: pd.DataFrame, df_full: pd.DataFrame) -> pd.DataFrame:
    """Prepare data for BN by merging original categoricals from full dataset."""
    # Get the BN features from the full (non-OHE) dataset using stay_id as key
    full_ids = set(df_full["stay_id"])
    ohe_ids = set(df_ohe["stay_id"])
    common_ids = full_ids & ohe_ids

    bn_feats = StrokeProfileBN.BN_FEATURES
    available = [c for c in bn_feats if c in df_full.columns]
    subset = df_full[df_full["stay_id"].isin(common_ids)][["stay_id"] + available].copy()
    # Drop duplicates on stay_id (some patients may have multiple rows)
    subset = subset.drop_duplicates(subset="stay_id")
    return subset


def _get_numeric_cols(df: pd.DataFrame) -> list:
    """Get numeric columns excluding IDs."""
    return [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in ID_COLS
    ]


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


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    results = {}
    t0 = time.time()

    # ==================================================================
    # 1. Load data
    # ==================================================================
    print("=" * 70)
    print("STEP 1: Loading data")
    print("=" * 70)

    train_ohe = pd.read_parquet(os.path.join(COHORT_DIR, "static_features_train.parquet"))
    test_ohe = pd.read_parquet(os.path.join(COHORT_DIR, "static_features_test.parquet"))
    full_static = pd.read_parquet(os.path.join(COHORT_DIR, "static_features.parquet"))
    ts_df = pd.read_parquet(os.path.join(COHORT_DIR, "timeseries_processed.parquet"))

    print(f"  Train (OHE):  {train_ohe.shape}")
    print(f"  Test  (OHE):  {test_ohe.shape}")
    print(f"  Full static:  {full_static.shape}")
    print(f"  Time-series:  {ts_df.shape}")

    n_test = len(test_ohe)

    # Prepare BN-compatible data (original categoricals)
    train_bn = _prepare_bn_data(train_ohe, full_static)
    test_bn = _prepare_bn_data(test_ohe, full_static)
    print(f"  BN train: {train_bn.shape}, BN test: {test_bn.shape}")

    # Add stroke_subtype and gender columns to OHE dataframes for evaluation
    train_ohe_eval = train_ohe.copy()
    test_ohe_eval = test_ohe.copy()
    train_ohe_eval["stroke_subtype"] = _reverse_ohe_stroke(train_ohe)
    test_ohe_eval["stroke_subtype"] = _reverse_ohe_stroke(test_ohe)
    train_ohe_eval["gender"] = _reverse_ohe_gender(train_ohe)
    test_ohe_eval["gender"] = _reverse_ohe_gender(test_ohe)

    # ==================================================================
    # 2. Fit models and generate synthetic data
    # ==================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Fitting models and generating synthetic data")
    print("=" * 70)

    # --- BN ---
    print("\n  [BN] Fitting Bayesian Network...")
    t1 = time.time()
    bn = StrokeProfileBN(max_indegree=3)
    bn.fit(train_bn)
    synth_bn_raw = bn.sample(n_test)
    bn_time = time.time() - t1
    print(f"  [BN] Done in {bn_time:.1f}s. Synthetic shape: {synth_bn_raw.shape}")
    print(f"  [BN] DAG edges: {len(bn.get_dag())}")

    # --- CTGAN ---
    print("\n  [CTGAN] Fitting CTGAN (epochs=50)...")
    t1 = time.time()
    # Use the OHE data for CTGAN (it handles mixed types)
    # Drop ID columns for training
    train_ctgan_cols = [c for c in train_ohe.columns if c not in ID_COLS]
    train_ctgan_df = train_ohe[train_ctgan_cols].copy()
    ctgan_meta = SingleTableMetadata()
    ctgan_meta.detect_from_dataframe(train_ctgan_df)
    ctgan_model = CTGANSynthesizer(
        ctgan_meta, epochs=50, batch_size=500, pac=1, verbose=False
    )
    ctgan_model.fit(train_ctgan_df)
    synth_ctgan = ctgan_model.sample(num_rows=n_test)
    ctgan_time = time.time() - t1
    print(f"  [CTGAN] Done in {ctgan_time:.1f}s. Synthetic shape: {synth_ctgan.shape}")

    # --- TVAE ---
    print("\n  [TVAE] Fitting TVAE (epochs=50)...")
    t1 = time.time()
    tvae_meta = SingleTableMetadata()
    tvae_meta.detect_from_dataframe(train_ctgan_df)
    tvae_model = TVAESynthesizer(
        tvae_meta, epochs=50, batch_size=500
    )
    tvae_model.fit(train_ctgan_df)
    synth_tvae = tvae_model.sample(num_rows=n_test)
    tvae_time = time.time() - t1
    print(f"  [TVAE] Done in {tvae_time:.1f}s. Synthetic shape: {synth_tvae.shape}")

    # Add stroke_subtype and gender to synthetic data for comparisons
    synth_ctgan_eval = synth_ctgan.copy()
    synth_tvae_eval = synth_tvae.copy()
    synth_ctgan_eval["stroke_subtype"] = _reverse_ohe_stroke(synth_ctgan)
    synth_tvae_eval["stroke_subtype"] = _reverse_ohe_stroke(synth_tvae)
    synth_ctgan_eval["gender"] = _reverse_ohe_gender(synth_ctgan)
    synth_tvae_eval["gender"] = _reverse_ohe_gender(synth_tvae)

    # For BN synthetic data: convert category dtypes to proper numeric/string
    synth_bn = synth_bn_raw.copy()
    for col in synth_bn.columns:
        if synth_bn[col].dtype.name == "category":
            # Try converting to numeric first
            try:
                synth_bn[col] = pd.to_numeric(synth_bn[col])
            except (ValueError, TypeError):
                synth_bn[col] = synth_bn[col].astype(str)

    results["model_training_times"] = {
        "bn_seconds": round(bn_time, 1),
        "ctgan_seconds": round(ctgan_time, 1),
        "tvae_seconds": round(tvae_time, 1),
    }

    # ==================================================================
    # 3. FIDELITY METRICS
    # ==================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Computing fidelity metrics")
    print("=" * 70)

    # Get common numeric columns for OHE-based comparisons
    real_test_numeric = test_ohe[_get_numeric_cols(test_ohe)]

    fidelity = {}
    for model_name, synth_df in [("BN", synth_bn), ("CTGAN", synth_ctgan), ("TVAE", synth_tvae)]:
        print(f"\n  [{model_name}] Computing fidelity metrics...")
        model_fidelity = {}

        # For BN, we compare on BN feature columns (numeric only)
        if model_name == "BN":
            bn_num_cols = [c for c in synth_bn.columns
                          if synth_bn[c].dtype in [np.float64, np.int64, np.float32, np.int32]
                          and c not in ID_COLS]
            real_for_bn = test_bn[[c for c in bn_num_cols if c in test_bn.columns]]
            synth_for_bn = synth_bn[[c for c in bn_num_cols if c in synth_bn.columns]]

            dwd = dimension_wise_distribution(real_for_bn, synth_for_bn)
            cp = correlation_preservation(real_for_bn, synth_for_bn)
            ds = discriminator_score(real_for_bn, synth_for_bn)
        else:
            synth_num = synth_df[_get_numeric_cols(synth_df)]
            dwd = dimension_wise_distribution(real_test_numeric, synth_num)
            cp = correlation_preservation(real_test_numeric, synth_num)
            ds = discriminator_score(real_test_numeric, synth_num)

        model_fidelity["avg_ks_pvalue"] = round(dwd["avg_pvalue"], 4)
        model_fidelity["frobenius_distance"] = round(cp["frobenius_distance"], 4)
        model_fidelity["discriminator_auc"] = round(ds["auc"], 4)
        model_fidelity["discriminator_auc_std"] = round(ds["auc_std"], 4)

        # Medical concept abundance for stroke_subtype
        if model_name == "BN":
            if "stroke_subtype" in synth_bn.columns and "stroke_subtype" in test_bn.columns:
                mca = medical_concept_abundance(test_bn, synth_bn, "stroke_subtype")
                model_fidelity["mca_manhattan"] = round(mca["manhattan_distance"], 4)
                model_fidelity["mca_real_dist"] = {str(k): round(v, 4) for k, v in mca["real_dist"].items()}
                model_fidelity["mca_synth_dist"] = {str(k): round(v, 4) for k, v in mca["synth_dist"].items()}
            else:
                model_fidelity["mca_manhattan"] = None
        else:
            eval_df = synth_ctgan_eval if model_name == "CTGAN" else synth_tvae_eval
            mca = medical_concept_abundance(test_ohe_eval, eval_df, "stroke_subtype")
            model_fidelity["mca_manhattan"] = round(mca["manhattan_distance"], 4)
            model_fidelity["mca_real_dist"] = {str(k): round(v, 4) for k, v in mca["real_dist"].items()}
            model_fidelity["mca_synth_dist"] = {str(k): round(v, 4) for k, v in mca["synth_dist"].items()}

        fidelity[model_name] = model_fidelity
        print(f"    KS avg p-value: {model_fidelity['avg_ks_pvalue']}")
        print(f"    Frobenius dist: {model_fidelity['frobenius_distance']}")
        print(f"    Disc AUC:       {model_fidelity['discriminator_auc']}")
        print(f"    MCA Manhattan:  {model_fidelity['mca_manhattan']}")

    results["fidelity"] = fidelity

    # ==================================================================
    # 4. CLINICAL PLAUSIBILITY
    # ==================================================================
    print("\n" + "=" * 70)
    print("STEP 4: Clinical plausibility rules")
    print("=" * 70)

    clinical = {}
    for label, df in [("Real", test_ohe_eval), ("BN", synth_bn), ("CTGAN", synth_ctgan_eval), ("TVAE", synth_tvae_eval)]:
        cr = check_clinical_rules(df)
        clinical[label] = {
            "total_violation_rate": round(cr["total_violation_rate"], 4),
            "total_violations": cr["total_violations"],
            "per_rule": {
                rule: {
                    "violations": info["violations"],
                    "violation_rate": round(info["violation_rate"], 4),
                }
                for rule, info in cr["per_rule"].items()
            },
        }
        print(f"  [{label}] Total violation rate: {cr['total_violation_rate']:.4f} ({cr['total_violations']} violations)")

    results["clinical_plausibility"] = clinical

    # ==================================================================
    # 5. UTILITY (TSTR)
    # ==================================================================
    print("\n" + "=" * 70)
    print("STEP 5: Utility (TSTR) evaluation")
    print("=" * 70)

    utility = {}
    train_cols = [c for c in train_ohe.columns if c not in ID_COLS]

    for model_name, synth_df in [("BN", None), ("CTGAN", synth_ctgan), ("TVAE", synth_tvae)]:
        print(f"\n  [{model_name}] Running TSTR...")
        if model_name == "BN":
            # BN synthetic data has different columns — need to map to OHE format
            # We'll use the BN features that overlap with train numeric columns
            bn_num = [c for c in synth_bn.columns
                      if c not in ID_COLS
                      and synth_bn[c].dtype in [np.float64, np.int64, np.float32, np.int32]
                      and c != "hospital_expire_flag"]
            common_feat = [c for c in bn_num if c in train_ohe.columns and c in test_ohe.columns]
            if "hospital_expire_flag" in synth_bn.columns and len(common_feat) > 0:
                bn_synth_for_tstr = synth_bn[common_feat + ["hospital_expire_flag"]].copy()
                bn_train_for_tstr = train_ohe[common_feat + ["hospital_expire_flag"]].copy()
                bn_test_for_tstr = test_ohe[common_feat + ["hospital_expire_flag"]].copy()
                tstr = tstr_evaluation(bn_train_for_tstr, bn_synth_for_tstr, bn_test_for_tstr)
            else:
                tstr = {"trtr_auc": 0, "tstr_auc": 0, "auc_gap": 0,
                        "lr_trtr_auc": 0, "lr_tstr_auc": 0, "rf_trtr_auc": 0, "rf_tstr_auc": 0}
        else:
            # Ensure hospital_expire_flag is present
            if "hospital_expire_flag" not in synth_df.columns:
                print(f"    WARNING: hospital_expire_flag not in {model_name} synthetic data")
                tstr = {"trtr_auc": 0, "tstr_auc": 0, "auc_gap": 0}
            else:
                try:
                    tstr = tstr_evaluation(
                        train_ohe[train_cols], synth_df, test_ohe[train_cols]
                    )
                except ValueError as e:
                    print(f"    WARNING: TSTR failed for {model_name}: {e}")
                    tstr = {"trtr_auc": 0, "tstr_auc": 0, "auc_gap": 0,
                            "lr_trtr_auc": 0, "lr_tstr_auc": 0,
                            "rf_trtr_auc": 0, "rf_tstr_auc": 0,
                            "error": str(e)}

        utility[model_name] = {k: round(v, 4) if isinstance(v, (int, float)) else v for k, v in tstr.items()}
        print(f"    TRTR AUC: {tstr.get('trtr_auc', 0):.4f}")
        print(f"    TSTR AUC: {tstr.get('tstr_auc', 0):.4f}")
        print(f"    Gap:      {tstr.get('auc_gap', 0):.4f}")

    results["utility"] = utility

    # ==================================================================
    # 6. PRIVACY METRICS
    # ==================================================================
    print("\n" + "=" * 70)
    print("STEP 6: Privacy metrics")
    print("=" * 70)

    privacy = {}
    quasi_ids_ohe = [c for c in _get_numeric_cols(train_ohe)
                     if c != "hospital_expire_flag"][:10]  # Top 10 numeric features

    for model_name, synth_df in [("BN", None), ("CTGAN", synth_ctgan), ("TVAE", synth_tvae)]:
        print(f"\n  [{model_name}] Computing privacy metrics...")
        model_privacy = {}

        if model_name == "BN":
            # Use BN-compatible numeric columns
            bn_num = [c for c in synth_bn.columns
                      if synth_bn[c].dtype in [np.float64, np.int64, np.float32, np.int32]
                      and c not in ID_COLS]
            common = [c for c in bn_num if c in train_ohe.columns]
            if common:
                real_priv = train_ohe[common]
                synth_priv = synth_bn[common]
                mia = membership_inference_attack(real_priv, synth_priv)
                nnd = nearest_neighbor_distance(real_priv, synth_priv)

                # AIA: predict hospital_expire_flag from other features
                qi = [c for c in common if c != "hospital_expire_flag"][:8]
                if "hospital_expire_flag" in synth_bn.columns:
                    aia = attribute_inference_attack(
                        train_ohe, synth_bn, "hospital_expire_flag", qi
                    )
                else:
                    aia = {"aia_accuracy": 0}
            else:
                mia = {"mia_f1": 0, "median_nn_distance": 0, "mean_nn_distance": 0}
                nnd = {"mean_dcr": 0, "median_dcr": 0, "min_dcr": 0, "p5_dcr": 0}
                aia = {"aia_accuracy": 0}
        else:
            mia = membership_inference_attack(train_ohe[_get_numeric_cols(train_ohe)],
                                              synth_df[_get_numeric_cols(synth_df)])
            nnd = nearest_neighbor_distance(train_ohe[_get_numeric_cols(train_ohe)],
                                            synth_df[_get_numeric_cols(synth_df)])
            # AIA: predict hospital_expire_flag
            qi = quasi_ids_ohe
            aia = attribute_inference_attack(
                train_ohe, synth_df, "hospital_expire_flag", qi
            )

        model_privacy["mia_f1"] = round(mia["mia_f1"], 4)
        model_privacy["median_nn_distance"] = round(mia.get("median_nn_distance", 0), 4)
        model_privacy["mean_dcr"] = round(nnd["mean_dcr"], 4)
        model_privacy["median_dcr"] = round(nnd["median_dcr"], 4)
        model_privacy["min_dcr"] = round(nnd["min_dcr"], 4)
        model_privacy["p5_dcr"] = round(nnd["p5_dcr"], 4)
        model_privacy["aia_accuracy"] = round(aia["aia_accuracy"], 4)

        privacy[model_name] = model_privacy
        print(f"    MIA F1:       {model_privacy['mia_f1']}")
        print(f"    Mean DCR:     {model_privacy['mean_dcr']}")
        print(f"    AIA Accuracy: {model_privacy['aia_accuracy']}")

    results["privacy"] = privacy

    # ==================================================================
    # 7. TEMPORAL METRICS (DGAN)
    # ==================================================================
    print("\n" + "=" * 70)
    print("STEP 7: Temporal evaluation (DGAN)")
    print("=" * 70)

    temporal = {}
    try:
        # Prepare time-series data for DGAN
        ts_features = ["hr", "sbp", "dbp", "map", "rr", "spo2", "temp_c"]
        meta_features = ["anchor_age", "hospital_expire_flag"]

        # Get unique stay_ids with enough timesteps
        stay_counts = ts_df.groupby("stay_id").size()
        valid_stays = stay_counts[stay_counts >= 24].index.tolist()
        print(f"  Stays with >=24 timesteps: {len(valid_stays)}")

        # Subsample to max 200 patients
        np.random.seed(42)
        if len(valid_stays) > 200:
            selected_stays = np.random.choice(valid_stays, 200, replace=False)
        else:
            selected_stays = valid_stays[:200]
        print(f"  Using {len(selected_stays)} patients for DGAN")

        # Build 3D array: (n_patients, seq_len=24, n_features)
        seq_len = 24
        sequences = []
        metadata_list = []

        # Get static metadata for selected stays
        full_static_map = full_static.set_index("stay_id")

        for sid in selected_stays:
            ts_patient = ts_df[ts_df["stay_id"] == sid].sort_values("hour")
            ts_vals = ts_patient[ts_features].fillna(method="ffill").fillna(0).values
            if len(ts_vals) < seq_len:
                continue
            ts_vals = ts_vals[:seq_len]
            sequences.append(ts_vals)

            # Get metadata
            if sid in full_static_map.index:
                row = full_static_map.loc[sid]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                meta = [float(row.get("anchor_age", 65)), float(row.get("hospital_expire_flag", 0))]
            else:
                meta = [65.0, 0.0]
            metadata_list.append(meta)

        sequences = np.array(sequences, dtype=np.float32)
        metadata_arr = np.array(metadata_list, dtype=np.float32)
        print(f"  Sequences shape: {sequences.shape}")
        print(f"  Metadata shape:  {metadata_arr.shape}")

        # Normalize sequences to [-1, 1]
        seq_min = sequences.min(axis=(0, 1), keepdims=True)
        seq_max = sequences.max(axis=(0, 1), keepdims=True)
        seq_range = seq_max - seq_min
        seq_range[seq_range == 0] = 1
        sequences_norm = 2 * (sequences - seq_min) / seq_range - 1

        # Normalize metadata
        meta_min = metadata_arr.min(axis=0, keepdims=True)
        meta_max = metadata_arr.max(axis=0, keepdims=True)
        meta_range = meta_max - meta_min
        meta_range[meta_range == 0] = 1
        metadata_norm = 2 * (metadata_arr - meta_min) / meta_range - 1

        # Fit DGAN
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

        # Generate synthetic sequences
        synth_ts = dgan.generate(metadata_norm)
        print(f"  Synthetic TS shape: {synth_ts.shape}")

        # DTW distance
        print("  Computing DTW distances...")
        dtw_res = dtw_distance_matrix(sequences_norm, synth_ts, n_samples=min(100, len(sequences_norm)))
        temporal["dtw"] = {k: round(v, 4) for k, v in dtw_res.items()}
        print(f"    Mean DTW: {dtw_res['mean_dtw']:.4f}")

        # Autocorrelation comparison
        print("  Computing autocorrelation comparison...")
        acf_res = autocorrelation_comparison(sequences_norm, synth_ts, ts_features, max_lag=12)
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
        results["model_training_times"]["dgan_seconds"] = round(dgan_time, 1)

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

    results_clean = _clean_for_json(results)
    json_path = os.path.join(OUTPUT_DIR, "evaluation_results.json")
    with open(json_path, "w") as f:
        json.dump(results_clean, f, indent=2, default=str)
    print(f"  Saved: {json_path}")

    # ==================================================================
    # 9. Generate CSV tables
    # ==================================================================

    # --- Table 4: Fidelity comparison ---
    table4_rows = []
    for model in ["BN", "CTGAN", "TVAE"]:
        f = fidelity[model]
        table4_rows.append({
            "Model": model,
            "Avg KS p-value": f["avg_ks_pvalue"],
            "Frobenius Distance": f["frobenius_distance"],
            "Discriminator AUC": f["discriminator_auc"],
            "MCA Manhattan": f["mca_manhattan"],
        })
    table4 = pd.DataFrame(table4_rows)
    table4_path = os.path.join(TABLES_DIR, "table4_fidelity_comparison.csv")
    table4.to_csv(table4_path, index=False)
    print(f"  Saved: {table4_path}")

    # --- Table 5: Clinical rules ---
    all_rules = list(next(iter(clinical.values()))["per_rule"].keys())
    table5_rows = []
    for label in ["Real", "BN", "CTGAN", "TVAE"]:
        row = {"Dataset": label, "Total Violation Rate": clinical[label]["total_violation_rate"]}
        for rule in all_rules:
            row[rule] = clinical[label]["per_rule"][rule]["violation_rate"]
        table5_rows.append(row)
    table5 = pd.DataFrame(table5_rows)
    table5_path = os.path.join(TABLES_DIR, "table5_clinical_rules.csv")
    table5.to_csv(table5_path, index=False)
    print(f"  Saved: {table5_path}")

    # --- Table 6: TSTR results ---
    table6_rows = []
    for model in ["BN", "CTGAN", "TVAE"]:
        u = utility[model]
        table6_rows.append({
            "Model": model,
            "LR TRTR AUC": u.get("lr_trtr_auc", ""),
            "LR TSTR AUC": u.get("lr_tstr_auc", ""),
            "RF TRTR AUC": u.get("rf_trtr_auc", ""),
            "RF TSTR AUC": u.get("rf_tstr_auc", ""),
            "Mean TRTR AUC": u.get("trtr_auc", ""),
            "Mean TSTR AUC": u.get("tstr_auc", ""),
            "AUC Gap": u.get("auc_gap", ""),
        })
    table6 = pd.DataFrame(table6_rows)
    table6_path = os.path.join(TABLES_DIR, "table6_tstr_results.csv")
    table6.to_csv(table6_path, index=False)
    print(f"  Saved: {table6_path}")

    # --- Table 7: Privacy metrics ---
    table7_rows = []
    for model in ["BN", "CTGAN", "TVAE"]:
        p = privacy[model]
        table7_rows.append({
            "Model": model,
            "MIA F1": p["mia_f1"],
            "Mean DCR": p["mean_dcr"],
            "Median DCR": p["median_dcr"],
            "5th %ile DCR": p["p5_dcr"],
            "AIA Accuracy": p["aia_accuracy"],
        })
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

    print(f"\nTotal runtime: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"Training samples: {len(train_ohe)}, Test samples: {len(test_ohe)}")

    print("\n--- FIDELITY (Table 4) ---")
    print(table4.to_string(index=False))

    print("\n--- CLINICAL RULES (Table 5) ---")
    print(table5[["Dataset", "Total Violation Rate"]].to_string(index=False))

    print("\n--- UTILITY / TSTR (Table 6) ---")
    print(table6.to_string(index=False))

    print("\n--- PRIVACY (Table 7) ---")
    print(table7.to_string(index=False))

    if "dtw" in temporal:
        print("\n--- TEMPORAL ---")
        print(f"  DTW: mean={temporal['dtw']['mean_dtw']}, median={temporal['dtw']['median_dtw']}")
        if "autocorrelation" in temporal:
            for feat, v in temporal["autocorrelation"].items():
                print(f"  ACF {feat}: mean_diff={v['mean_diff']}, max_diff={v['max_diff']}")

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
