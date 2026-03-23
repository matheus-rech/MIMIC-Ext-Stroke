"""
Drift Detection -- Monitor divergence between virtual predictions and physical observations.

Uses EMA (Exponential Moving Average) with alpha=0.3 for responsiveness to
recent distribution shifts.  Flags recalibration when drift exceeds clinically
meaningful thresholds for stroke monitoring channels.

Two detection methods:

  1. compute_drift_report -- EMA-based drift summary from TwinState feedback.
     Fast, online, per-patient.  Threshold: recent_drift > 0.5 triggers
     recalibration flag.

  2. detect_distribution_shift -- KS test between real and predicted windows.
     Statistical test for distributional divergence.  Per-channel p-values
     identify which physiological signals have shifted.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

# Default recalibration threshold for the EMA drift metric.
# Value 0.5 on normalized [-1, 1] scale corresponds to a ~25% deviation
# from the center of the output range -- large enough to indicate
# systematic model-reality divergence rather than noise.
DEFAULT_DRIFT_THRESHOLD = 0.5


def compute_drift_report(
    state: "TwinState",  # noqa: F821 -- forward ref to avoid circular import
    threshold: float = DEFAULT_DRIFT_THRESHOLD,
) -> dict:
    """Compute drift metrics from a TwinState's feedback history.

    Aggregates per-channel MAE across all feedback records and checks the
    EMA drift against the recalibration threshold.

    Args:
        state: TwinState with feedback records.
        threshold: EMA drift value above which needs_recalibration is True.
                   Default 0.5 (on normalized scale).

    Returns:
        Dict with keys:
            overall_mae:          float, mean absolute error across all feedback.
            recent_drift_ema:     float, current EMA value from state.
            per_channel_mae:      dict[str, float], per-channel average MAE.
            needs_recalibration:  bool, True if recent_drift > threshold.
            n_feedback_records:   int, number of feedback entries evaluated.
    """
    if not state.feedback:
        return {
            "overall_mae": 0.0,
            "recent_drift_ema": state.recent_drift,
            "per_channel_mae": {},
            "needs_recalibration": False,
            "n_feedback_records": 0,
        }

    # Accumulate per-channel errors across all feedback records
    channel_errors: Dict[str, List[float]] = {}
    all_errors: List[float] = []

    for fb in state.feedback:
        for ch, err in fb.error.items():
            channel_errors.setdefault(ch, []).append(err)
            all_errors.append(err)

    overall_mae = float(np.mean(all_errors)) if all_errors else 0.0
    per_channel_mae = {
        ch: float(np.mean(errs)) for ch, errs in channel_errors.items()
    }

    needs_recal = state.recent_drift > threshold

    report = {
        "overall_mae": overall_mae,
        "recent_drift_ema": state.recent_drift,
        "per_channel_mae": per_channel_mae,
        "needs_recalibration": needs_recal,
        "n_feedback_records": len(state.feedback),
    }

    if needs_recal:
        logger.warning(
            "DRIFT ALERT: entity=%s, EMA=%.4f > threshold=%.4f. "
            "Recalibration recommended.",
            state.entity_id,
            state.recent_drift,
            threshold,
        )

    return report


def detect_distribution_shift(
    real_window: np.ndarray,
    predicted_window: np.ndarray,
    channel_names: Optional[List[str]] = None,
) -> dict:
    """KS test between recent real observations and model predictions.

    Performs a two-sample Kolmogorov-Smirnov test per channel to detect
    distributional divergence between what the model predicts and what
    is actually observed.  Useful for detecting covariate shift, concept
    drift, or model staleness.

    Args:
        real_window: Recent real observations, shape (n_real, n_channels).
        predicted_window: Recent model predictions, shape (n_pred, n_channels).
        channel_names: Optional list of channel names for labeling.
                       If None, channels are labeled by index.

    Returns:
        Dict with keys:
            per_channel: dict[str, dict] with 'ks_statistic' and 'p_value'.
            any_significant: bool, True if any channel p < 0.05.
            n_real: int, number of real observations used.
            n_predicted: int, number of predictions used.
    """
    if real_window.ndim == 1:
        real_window = real_window[:, np.newaxis]
    if predicted_window.ndim == 1:
        predicted_window = predicted_window[:, np.newaxis]

    n_channels = real_window.shape[1]
    if channel_names is None:
        channel_names = [f"ch_{i}" for i in range(n_channels)]

    per_channel = {}
    any_significant = False

    for i, ch_name in enumerate(channel_names):
        real_col = real_window[:, i]
        pred_col = predicted_window[:, i]

        # Filter out NaNs
        real_valid = real_col[~np.isnan(real_col)]
        pred_valid = pred_col[~np.isnan(pred_col)]

        if len(real_valid) < 2 or len(pred_valid) < 2:
            per_channel[ch_name] = {
                "ks_statistic": float("nan"),
                "p_value": float("nan"),
                "note": "insufficient data",
            }
            continue

        ks_stat, p_val = sp_stats.ks_2samp(real_valid, pred_valid)
        per_channel[ch_name] = {
            "ks_statistic": float(ks_stat),
            "p_value": float(p_val),
        }

        if p_val < 0.05:
            any_significant = True
            logger.info(
                "Distribution shift detected: %s (KS=%.4f, p=%.4f)",
                ch_name,
                ks_stat,
                p_val,
            )

    return {
        "per_channel": per_channel,
        "any_significant": any_significant,
        "n_real": real_window.shape[0],
        "n_predicted": predicted_window.shape[0],
    }
