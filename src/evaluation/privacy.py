"""Privacy evaluation metrics for synthetic data."""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler


def membership_inference_attack(real: pd.DataFrame, synth: pd.DataFrame, k: int = 5) -> dict:
    """Distance-based membership inference attack.

    For each synthetic record, find k nearest neighbors in real data.
    If distances are very small, the synthetic record may have memorized a real patient.
    """
    numeric_cols = [
        c for c in real.select_dtypes(include=[np.number]).columns if c in synth.columns
    ]

    X_real = real[numeric_cols].fillna(0).values
    X_synth = synth[numeric_cols].fillna(0).values

    scaler = StandardScaler()
    X_real_s = scaler.fit_transform(X_real)
    X_synth_s = scaler.transform(X_synth)

    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X_real_s)
    distances, _ = nn.kneighbors(X_synth_s)

    # Threshold: if nearest neighbor distance < median distance, classify as "member"
    median_dist = np.median(distances[:, 0])

    # Mix real and synthetic, try to classify
    X_mixed = np.vstack([X_real_s[: len(X_synth_s)], X_synth_s])
    y_mixed = np.concatenate(
        [np.ones(min(len(X_real_s), len(X_synth_s))), np.zeros(len(X_synth_s))]
    )

    nn2 = NearestNeighbors(n_neighbors=1)
    nn2.fit(X_real_s)
    dists, _ = nn2.kneighbors(X_mixed)
    threshold = np.median(dists[:, 0])
    preds = (dists[:, 0] < threshold).astype(int)

    mia_f1 = f1_score(y_mixed, preds, zero_division=0)

    return {
        "mia_f1": mia_f1,
        "median_nn_distance": float(median_dist),
        "mean_nn_distance": float(distances[:, 0].mean()),
    }


def nearest_neighbor_distance(real: pd.DataFrame, synth: pd.DataFrame) -> dict:
    """Distance to Closest Record (DCR) for privacy assessment."""
    numeric_cols = [
        c for c in real.select_dtypes(include=[np.number]).columns if c in synth.columns
    ]

    X_real = real[numeric_cols].fillna(0).values
    X_synth = synth[numeric_cols].fillna(0).values

    scaler = StandardScaler()
    X_real_s = scaler.fit_transform(X_real)
    X_synth_s = scaler.transform(X_synth)

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(X_real_s)
    distances, indices = nn.kneighbors(X_synth_s)

    return {
        "mean_dcr": float(distances.mean()),
        "median_dcr": float(np.median(distances)),
        "min_dcr": float(distances.min()),
        "p5_dcr": float(np.percentile(distances, 5)),
    }


def attribute_inference_attack(
    real: pd.DataFrame, synth: pd.DataFrame, sensitive_col: str, quasi_ids: list
) -> dict:
    """Attempt to infer sensitive attribute from quasi-identifiers."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    common_cols = [c for c in quasi_ids if c in real.columns and c in synth.columns]
    if sensitive_col not in real.columns or sensitive_col not in synth.columns:
        return {"aia_accuracy": 0, "error": "sensitive column not found"}

    X_synth = synth[common_cols].fillna(0).values
    y_synth = synth[sensitive_col].values
    X_real = real[common_cols].fillna(0).values
    y_real = real[sensitive_col].values

    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_synth, y_synth)
    y_pred = clf.predict(X_real)

    return {"aia_accuracy": float(accuracy_score(y_real, y_pred))}
