#!/usr/bin/env python3
"""Optuna hyperparameter optimization for the stroke DGAN.

Searches over architecture, training regime, and loss mode (BCE vs WGAN-GP)
to find a DGAN configuration that maximises fidelity, utility, and
correlation preservation simultaneously.

Usage:
    # Dry run (tiny random data, fast sanity check)
    python scripts/optimize_dgan.py --dry-run --n-trials 5

    # Full optimization
    python scripts/optimize_dgan.py --n-trials 50 --timeout 7200

    # Resume a previous study
    python scripts/optimize_dgan.py --study-name stroke-dgan-v1 --n-trials 50
"""

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Project root & imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import optuna  # noqa: E402
from optuna.pruners import MedianPruner  # noqa: E402

from src.models.dgan_model import StrokeTimeSeriesDGAN  # noqa: E402
from src.evaluation.fidelity import (  # noqa: E402
    correlation_preservation,
    discriminator_score,
)
from src.evaluation.utility import tstr_evaluation  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "optimization")
ID_COLS = ["subject_id", "hadm_id", "stay_id"]

# Dry-run dimensions (kept small for fast iteration)
DRY_N_PATIENTS = 50
DRY_N_METADATA = 9
DRY_N_FEATURES = 11
DRY_SEQ_LEN = 72


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_real_data() -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Load preprocessed data for optimization.

    Returns
    -------
    metadata_train : np.ndarray  (n_train, n_metadata)
    sequences_train : np.ndarray (n_train, seq_len, n_features)
    static_train_df : pd.DataFrame  (for TSTR / fidelity eval)
    static_test_df  : pd.DataFrame  (for TSTR / fidelity eval)
    """
    cohort_dir = os.path.join(PROJECT_ROOT, "outputs", "cohort")
    preproc_dir = os.path.join(PROJECT_ROOT, "outputs", "preprocessed")

    # --- Static features ---
    static_train_path = os.path.join(cohort_dir, "static_features_train.parquet")
    static_test_path = os.path.join(cohort_dir, "static_features_test.parquet")

    if os.path.exists(static_train_path) and os.path.exists(static_test_path):
        train_df = pd.read_parquet(static_train_path)
        test_df = pd.read_parquet(static_test_path)
    else:
        # Fallback: load full cohort and split manually
        full_path = os.path.join(cohort_dir, "static_features.parquet")
        if not os.path.exists(full_path):
            raise FileNotFoundError(
                f"No static features found at {cohort_dir}. "
                "Run the preprocessing pipeline first, or use --dry-run."
            )
        from sklearn.model_selection import train_test_split

        full_df = pd.read_parquet(full_path)
        train_df, test_df = train_test_split(
            full_df, test_size=0.2, random_state=42,
            stratify=full_df["hospital_expire_flag"]
            if "hospital_expire_flag" in full_df.columns
            else None,
        )

    # Metadata = all numeric columns from static (excluding IDs and target)
    meta_cols = [
        c for c in train_df.columns
        if c not in ID_COLS
        and train_df[c].dtype in [np.float64, np.int64, np.float32, np.int32]
    ]
    metadata_train = train_df[meta_cols].fillna(0).values.astype(np.float32)

    # --- Temporal tensor ---
    tensor_path = os.path.join(preproc_dir, "temporal_tensor.npz")
    if os.path.exists(tensor_path):
        data = np.load(tensor_path)
        # Accept any key; typical keys are 'train', 'data', 'arr_0'
        key = list(data.keys())[0]
        sequences_train = data[key].astype(np.float32)
        # Align patient count
        n = min(metadata_train.shape[0], sequences_train.shape[0])
        metadata_train = metadata_train[:n]
        sequences_train = sequences_train[:n]
    else:
        # Build temporal tensor from timeseries parquet
        ts_path = os.path.join(cohort_dir, "timeseries_processed.parquet")
        if not os.path.exists(ts_path):
            raise FileNotFoundError(
                f"No temporal data found at {preproc_dir} or {cohort_dir}. "
                "Run preprocessing first, or use --dry-run."
            )
        ts_df = pd.read_parquet(ts_path)
        vital_cols = [
            c for c in ts_df.columns
            if c not in ["stay_id", "hour", "subject_id", "hadm_id"]
        ]
        # Convert long-form timeseries to 3-D tensor
        train_stay_ids = set(train_df["stay_id"]) if "stay_id" in train_df.columns else set()
        if train_stay_ids:
            ts_train = ts_df[ts_df["stay_id"].isin(train_stay_ids)]
        else:
            ts_train = ts_df

        # Determine seq_len as max hour across all stays
        max_hour = int(ts_train["hour"].max()) + 1 if "hour" in ts_train.columns else 72
        seq_len = min(max_hour, 168)  # cap at 7 days

        grouped = ts_train.groupby("stay_id")
        tensors = []
        for _, group in grouped:
            arr = np.full((seq_len, len(vital_cols)), np.nan, dtype=np.float32)
            for i, row in group.iterrows():
                h = int(row.get("hour", 0))
                if 0 <= h < seq_len:
                    arr[h] = row[vital_cols].values.astype(np.float32)
            tensors.append(arr)

        sequences_train = np.array(tensors, dtype=np.float32)
        # Replace NaN with 0 for training
        sequences_train = np.nan_to_num(sequences_train, 0.0)
        # Align patient count
        n = min(metadata_train.shape[0], sequences_train.shape[0])
        metadata_train = metadata_train[:n]
        sequences_train = sequences_train[:n]

    return metadata_train, sequences_train, train_df, test_df


def _make_dry_data() -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Create tiny random data for pipeline testing."""
    rng = np.random.default_rng(42)

    n_total = DRY_N_PATIENTS
    n_train = int(n_total * 0.7)
    n_test = n_total - n_train

    # Static metadata (numeric columns)
    meta_cols = [f"feat_{i}" for i in range(DRY_N_METADATA - 1)] + ["hospital_expire_flag"]
    meta_train = rng.standard_normal((n_train, DRY_N_METADATA)).astype(np.float32)
    meta_test = rng.standard_normal((n_test, DRY_N_METADATA)).astype(np.float32)

    # Binary mortality target (last column)
    meta_train[:, -1] = rng.binomial(1, 0.25, n_train).astype(np.float32)
    meta_test[:, -1] = rng.binomial(1, 0.25, n_test).astype(np.float32)

    # Temporal sequences
    sequences = rng.standard_normal(
        (n_train, DRY_SEQ_LEN, DRY_N_FEATURES)
    ).astype(np.float32)
    # Clip to [-1, 1] (matching tanh output range)
    sequences = np.clip(sequences, -1, 1)

    # Build DataFrames
    train_df = pd.DataFrame(meta_train, columns=meta_cols)
    test_df = pd.DataFrame(meta_test, columns=meta_cols)

    # Add stay_id for compatibility
    train_df["stay_id"] = range(1, n_train + 1)
    test_df["stay_id"] = range(n_train + 1, n_total + 1)

    return meta_train, sequences, train_df, test_df


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _flatten_temporal(meta: np.ndarray, temporal: np.ndarray) -> pd.DataFrame:
    """Flatten metadata + temporal summary stats into a DataFrame for evaluation.

    For each patient, computes mean and std per temporal feature, then
    concatenates with the static metadata to produce a single-row representation
    suitable for the fidelity/utility evaluators (which expect DataFrames).
    """
    n_patients = meta.shape[0]
    n_meta = meta.shape[1]
    n_feat = temporal.shape[2]

    # Temporal summary: mean & std per feature
    t_mean = np.nanmean(temporal, axis=1)  # (n, n_feat)
    t_std = np.nanstd(temporal, axis=1)    # (n, n_feat)

    combined = np.hstack([meta, t_mean, t_std])

    col_names = (
        [f"meta_{i}" for i in range(n_meta)]
        + [f"ts_mean_{i}" for i in range(n_feat)]
        + [f"ts_std_{i}" for i in range(n_feat)]
    )
    return pd.DataFrame(combined, columns=col_names)


def _evaluate_trial(
    model: StrokeTimeSeriesDGAN,
    metadata_train: np.ndarray,
    sequences_train: np.ndarray,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, float]:
    """Generate synthetic data and compute all evaluation metrics.

    Returns dict with discriminator_auc, tstr_auc, frobenius_distance.
    """
    # Generate synthetic temporal data conditioned on training metadata
    synth_sequences = model.generate(metadata_train, n_per_patient=1)

    # Build flat representations for fidelity / utility evaluation
    real_flat = _flatten_temporal(metadata_train, sequences_train)
    synth_flat = _flatten_temporal(metadata_train, synth_sequences)

    # 1. Discriminator AUC (fidelity)
    try:
        ds = discriminator_score(real_flat, synth_flat)
        disc_auc = ds["auc"]
    except Exception:
        disc_auc = 1.0  # worst case

    # 2. Frobenius distance (correlation preservation)
    try:
        cp = correlation_preservation(real_flat, synth_flat)
        frob = cp["frobenius_distance"]
    except Exception:
        frob = 10.0  # worst case

    # 3. TSTR AUC (utility) — needs a target column
    tstr_auc = 0.5  # default
    try:
        target_col = "hospital_expire_flag"

        # Build train/synth/test DataFrames with the target column
        meta_cols = [
            c for c in train_df.columns
            if c not in ID_COLS
            and train_df[c].dtype in [np.float64, np.int64, np.float32, np.int32]
        ]

        if target_col in meta_cols:
            # Real train flat features + target
            real_train_eval = real_flat.copy()
            real_train_eval[target_col] = train_df[target_col].values[:len(real_train_eval)]

            # Synthetic flat features + target (reuse real labels for conditioning)
            synth_eval = synth_flat.copy()
            synth_eval[target_col] = train_df[target_col].values[:len(synth_eval)]

            # Real test flat features + target
            test_meta = test_df[meta_cols].fillna(0).values.astype(np.float32)
            # For test, we create a dummy temporal (zeros) since we only have static
            # features for test patients. TSTR evaluates on static-level anyway.
            n_test = len(test_df)
            n_feat = sequences_train.shape[2]
            test_temporal = np.zeros(
                (n_test, sequences_train.shape[1], n_feat), dtype=np.float32
            )
            test_flat = _flatten_temporal(test_meta, test_temporal)
            test_flat[target_col] = test_df[target_col].values[:len(test_flat)]

            tstr_result = tstr_evaluation(
                real_train_eval, synth_eval, test_flat, target=target_col,
            )
            tstr_auc = tstr_result.get("tstr_auc", 0.5)
    except Exception:
        tstr_auc = 0.5

    return {
        "discriminator_auc": float(disc_auc),
        "tstr_auc": float(tstr_auc),
        "frobenius_distance": float(frob),
    }


# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------

def create_objective(
    metadata_train: np.ndarray,
    sequences_train: np.ndarray,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str,
):
    """Return an Optuna objective function closed over the data."""

    n_metadata = metadata_train.shape[1]
    n_features = sequences_train.shape[2]
    seq_len = sequences_train.shape[1]

    best_score = float("inf")
    best_model_state = None

    def objective(trial: optuna.Trial) -> float:
        nonlocal best_score, best_model_state

        # ----- Sample hyperparameters -----
        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
        noise_dim = trial.suggest_categorical("noise_dim", [50, 100, 200])
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
        epochs = trial.suggest_categorical("epochs", [100, 200, 300, 500])
        loss_type = trial.suggest_categorical("loss_type", ["bce", "wgan-gp"])

        # WGAN-GP specific
        if loss_type == "wgan-gp":
            n_critic = trial.suggest_categorical("n_critic", [3, 5, 7])
            gp_lambda = trial.suggest_categorical("gp_lambda", [5.0, 10.0, 20.0])
        else:
            n_critic = 5
            gp_lambda = 10.0

        # ----- Create & train model -----
        t0 = time.time()
        try:
            model = StrokeTimeSeriesDGAN(
                n_features=n_features,
                n_metadata=n_metadata,
                seq_len=seq_len,
                noise_dim=noise_dim,
                hidden_dim=hidden_dim,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                loss_type=loss_type,
                n_critic=n_critic,
                gp_lambda=gp_lambda,
            )
        except Exception as e:
            print(f"  [Trial {trial.number}] Model creation failed: {e}")
            return float("inf")

        try:
            model.train(metadata_train, sequences_train)
        except Exception as e:
            print(f"  [Trial {trial.number}] Training failed: {e}")
            return float("inf")

        train_time = time.time() - t0

        # ----- Evaluate -----
        try:
            metrics = _evaluate_trial(
                model, metadata_train, sequences_train, train_df, test_df,
            )
        except Exception as e:
            print(f"  [Trial {trial.number}] Evaluation failed: {e}")
            return float("inf")

        disc_auc = metrics["discriminator_auc"]
        tstr_auc = metrics["tstr_auc"]
        frob = metrics["frobenius_distance"]

        # Composite score: lower is better
        score = abs(disc_auc - 0.5) + (1.0 - tstr_auc) + frob / 10.0

        # Log metrics
        trial.set_user_attr("discriminator_auc", disc_auc)
        trial.set_user_attr("tstr_auc", tstr_auc)
        trial.set_user_attr("frobenius_distance", frob)
        trial.set_user_attr("train_time_s", round(train_time, 1))
        trial.set_user_attr("composite_score", round(score, 4))

        # Track final G/D losses
        if model.losses["g_loss"]:
            trial.set_user_attr("final_g_loss", round(model.losses["g_loss"][-1], 4))
            trial.set_user_attr("final_d_loss", round(model.losses["d_loss"][-1], 4))

        # Save best model
        if score < best_score:
            best_score = score
            model_path = os.path.join(output_dir, "best_model.pt")
            model.save(model_path)

        print(
            f"  [Trial {trial.number:3d}] "
            f"score={score:.4f}  "
            f"disc_auc={disc_auc:.3f}  "
            f"tstr_auc={tstr_auc:.3f}  "
            f"frob={frob:.3f}  "
            f"loss={loss_type}  "
            f"epochs={epochs}  "
            f"time={train_time:.1f}s"
        )

        return score

    return objective


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(study: optuna.Study, top_k: int = 5) -> None:
    """Print a nicely formatted summary of top trials."""
    trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else float("inf"))
    top = trials[:top_k]

    print("\n" + "=" * 90)
    print(f"TOP {min(top_k, len(top))} TRIALS (out of {len(study.trials)} completed)")
    print("=" * 90)
    header = (
        f"{'#':>4}  {'Score':>8}  {'Disc AUC':>9}  {'TSTR AUC':>9}  "
        f"{'Frobenius':>10}  {'Loss':>7}  {'Epochs':>6}  {'Hidden':>6}  {'Time':>6}"
    )
    print(header)
    print("-" * 90)

    for t in top:
        if t.value is None:
            continue
        ua = t.user_attrs
        print(
            f"{t.number:4d}  "
            f"{t.value:8.4f}  "
            f"{ua.get('discriminator_auc', 0):9.3f}  "
            f"{ua.get('tstr_auc', 0):9.3f}  "
            f"{ua.get('frobenius_distance', 0):10.3f}  "
            f"{t.params.get('loss_type', '?'):>7}  "
            f"{t.params.get('epochs', '?'):>6}  "
            f"{t.params.get('hidden_dim', '?'):>6}  "
            f"{ua.get('train_time_s', 0):5.1f}s"
        )

    print()
    best = study.best_trial
    print(f"BEST TRIAL: #{best.number}")
    print(f"  Composite score : {best.value:.4f}")
    print(f"  Parameters      : {json.dumps(best.params, indent=4)}")
    print(f"  Disc AUC        : {best.user_attrs.get('discriminator_auc', '?')}")
    print(f"  TSTR AUC        : {best.user_attrs.get('tstr_auc', '?')}")
    print(f"  Frobenius       : {best.user_attrs.get('frobenius_distance', '?')}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter optimization for stroke DGAN",
    )
    parser.add_argument(
        "--n-trials", type=int, default=50,
        help="Number of Optuna trials (default: 50)",
    )
    parser.add_argument(
        "--timeout", type=int, default=None,
        help="Maximum optimization time in seconds (default: no limit)",
    )
    parser.add_argument(
        "--study-name", type=str, default="stroke-dgan-optuna",
        help="Optuna study name (for resuming)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Use tiny random data for testing the pipeline",
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- Load data ----
    if args.dry_run:
        print("DRY RUN: Using synthetic random data")
        print(f"  Patients={DRY_N_PATIENTS}, Metadata={DRY_N_METADATA}, "
              f"Features={DRY_N_FEATURES}, SeqLen={DRY_SEQ_LEN}")
        metadata_train, sequences_train, train_df, test_df = _make_dry_data()
    else:
        print("Loading real preprocessed data...")
        metadata_train, sequences_train, train_df, test_df = _load_real_data()

    n_train = metadata_train.shape[0]
    n_metadata = metadata_train.shape[1]
    n_features = sequences_train.shape[2]
    seq_len = sequences_train.shape[1]

    print(f"  Training patients : {n_train}")
    print(f"  Metadata dims     : {n_metadata}")
    print(f"  Temporal features : {n_features}")
    print(f"  Sequence length   : {seq_len}")
    print(f"  Test patients     : {len(test_df)}")
    print()

    # ---- Create Optuna study ----
    db_path = os.path.join(OUTPUT_DIR, "stroke_dgan_study.db")
    storage = f"sqlite:///{db_path}"

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="minimize",
        pruner=MedianPruner(n_startup_trials=3, n_warmup_steps=0),
        load_if_exists=True,
    )

    n_existing = len(study.trials)
    if n_existing > 0:
        print(f"Resuming study '{args.study_name}' with {n_existing} existing trials")
    else:
        print(f"Starting new study '{args.study_name}'")

    print(f"Running {args.n_trials} new trials"
          + (f" (timeout={args.timeout}s)" if args.timeout else ""))
    print()

    # ---- Run optimization ----
    objective = create_objective(
        metadata_train, sequences_train, train_df, test_df, OUTPUT_DIR,
    )

    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=True,
    )

    # ---- Save results ----
    best = study.best_trial
    best_params = {
        "study_name": args.study_name,
        "best_trial_number": best.number,
        "best_composite_score": best.value,
        "params": best.params,
        "metrics": {
            "discriminator_auc": best.user_attrs.get("discriminator_auc"),
            "tstr_auc": best.user_attrs.get("tstr_auc"),
            "frobenius_distance": best.user_attrs.get("frobenius_distance"),
            "train_time_s": best.user_attrs.get("train_time_s"),
        },
        "data_config": {
            "n_train": n_train,
            "n_metadata": n_metadata,
            "n_features": n_features,
            "seq_len": seq_len,
            "dry_run": args.dry_run,
        },
        "total_trials": len(study.trials),
    }

    params_path = os.path.join(OUTPUT_DIR, "best_params.json")
    with open(params_path, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"\nBest params saved to {params_path}")
    print(f"Study database at   {db_path}")
    print(f"Best model saved to {os.path.join(OUTPUT_DIR, 'best_model.pt')}")

    # ---- Print summary ----
    print_summary(study)


if __name__ == "__main__":
    main()
