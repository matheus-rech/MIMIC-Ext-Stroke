"""Statistical fidelity metrics for synthetic EHR evaluation.

Implements metrics from the 12-metric framework (Yan et al., Nature Comms 2022).
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


def dimension_wise_distribution(real: pd.DataFrame, synth: pd.DataFrame) -> dict:
    """KS test per numeric column."""
    numeric_cols = real.select_dtypes(include=[np.number]).columns
    common_cols = [c for c in numeric_cols if c in synth.columns]

    results = {}
    pvalues = []
    for col in common_cols:
        r = real[col].dropna()
        s = synth[col].dropna()
        if len(r) > 0 and len(s) > 0:
            stat, pval = ks_2samp(r, s)
            results[col] = {"ks_stat": stat, "pvalue": pval}
            pvalues.append(pval)

    return {"per_column": results, "avg_pvalue": np.mean(pvalues) if pvalues else 0}


def correlation_preservation(real: pd.DataFrame, synth: pd.DataFrame) -> dict:
    """Compare Pearson correlation matrices via Frobenius distance."""
    numeric_cols = real.select_dtypes(include=[np.number]).columns
    common_cols = [c for c in numeric_cols if c in synth.columns]

    real_corr = real[common_cols].corr().fillna(0)
    synth_corr = synth[common_cols].corr().fillna(0)

    diff = real_corr - synth_corr
    frobenius = np.sqrt((diff.values**2).sum())

    return {
        "frobenius_distance": frobenius,
        "real_corr": real_corr,
        "synth_corr": synth_corr,
    }


def discriminator_score(real: pd.DataFrame, synth: pd.DataFrame) -> dict:
    """Train classifier to distinguish real vs synthetic; return AUC."""
    numeric_cols = real.select_dtypes(include=[np.number]).columns
    common_cols = [c for c in numeric_cols if c in synth.columns]

    X_real = real[common_cols].fillna(0).values
    X_synth = synth[common_cols].fillna(0).values

    X = np.vstack([X_real, X_synth])
    y = np.concatenate([np.ones(len(X_real)), np.zeros(len(X_synth))])

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    scores = cross_val_score(clf, X, y, cv=5, scoring="roc_auc")

    return {"auc": scores.mean(), "auc_std": scores.std()}


def medical_concept_abundance(real: pd.DataFrame, synth: pd.DataFrame, column: str) -> dict:
    """Normalized Manhattan distance of categorical distributions."""
    real_dist = real[column].value_counts(normalize=True).sort_index()
    synth_dist = synth[column].value_counts(normalize=True).sort_index()

    all_cats = sorted(set(real_dist.index) | set(synth_dist.index))
    real_vec = np.array([real_dist.get(c, 0) for c in all_cats])
    synth_vec = np.array([synth_dist.get(c, 0) for c in all_cats])

    manhattan = np.abs(real_vec - synth_vec).sum()

    return {
        "manhattan_distance": manhattan,
        "real_dist": dict(zip(all_cats, real_vec)),
        "synth_dist": dict(zip(all_cats, synth_vec)),
    }
