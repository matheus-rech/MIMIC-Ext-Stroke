"""Downstream utility evaluation via TSTR/TRTS.

Train on Synthetic, Test on Real (TSTR) measures whether synthetic data
preserves enough signal for downstream ML tasks.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


def tstr_evaluation(
    real_train: pd.DataFrame,
    synth: pd.DataFrame,
    real_test: pd.DataFrame,
    target: str = "hospital_expire_flag",
) -> dict:
    """TSTR and TRTR evaluation for mortality prediction."""
    feature_cols = [
        c
        for c in real_train.columns
        if c != target and real_train[c].dtype in [np.float64, np.int64, np.float32, np.int32]
    ]

    X_real_train = real_train[feature_cols].fillna(0).values
    y_real_train = real_train[target].values
    X_synth = synth[feature_cols].fillna(0).values
    y_synth = synth[target].values
    X_test = real_test[feature_cols].fillna(0).values
    y_test = real_test[target].values

    scaler = StandardScaler()
    X_real_train_s = scaler.fit_transform(X_real_train)
    X_test_s = scaler.transform(X_test)

    scaler_synth = StandardScaler()
    X_synth_s = scaler_synth.fit_transform(X_synth)
    X_test_synth_s = scaler_synth.transform(X_test)

    results = {}
    for name, clf_cls in [("lr", LogisticRegression), ("rf", RandomForestClassifier)]:
        # TRTR: Train Real, Test Real
        clf_real = (
            clf_cls(max_iter=1000, random_state=42)
            if name == "lr"
            else clf_cls(n_estimators=100, random_state=42)
        )
        clf_real.fit(X_real_train_s, y_real_train)
        y_pred_real = (
            clf_real.predict_proba(X_test_s)[:, 1]
            if hasattr(clf_real, "predict_proba")
            else clf_real.predict(X_test_s)
        )

        # TSTR: Train Synthetic, Test Real
        clf_synth = (
            clf_cls(max_iter=1000, random_state=42)
            if name == "lr"
            else clf_cls(n_estimators=100, random_state=42)
        )
        clf_synth.fit(X_synth_s, y_synth)
        y_pred_synth = (
            clf_synth.predict_proba(X_test_synth_s)[:, 1]
            if hasattr(clf_synth, "predict_proba")
            else clf_synth.predict(X_test_synth_s)
        )

        try:
            trtr_auc = roc_auc_score(y_test, y_pred_real)
            tstr_auc = roc_auc_score(y_test, y_pred_synth)
        except ValueError:
            trtr_auc = tstr_auc = 0.5

        results[f"{name}_trtr_auc"] = trtr_auc
        results[f"{name}_tstr_auc"] = tstr_auc
        results[f"{name}_gap"] = abs(trtr_auc - tstr_auc)

    # Summary
    results["trtr_auc"] = np.mean([results["lr_trtr_auc"], results["rf_trtr_auc"]])
    results["tstr_auc"] = np.mean([results["lr_tstr_auc"], results["rf_tstr_auc"]])
    results["auc_gap"] = abs(results["trtr_auc"] - results["tstr_auc"])

    return results
