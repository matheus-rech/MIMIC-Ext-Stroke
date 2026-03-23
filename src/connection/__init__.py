"""
Connection Layer -- Grieves Triad Bidirectional Connection for Stroke Digital Twins.

The Grieves digital twin triad requires three components:
    1. Physical entity  (real stroke patients in the ICU)      -- exists
    2. Virtual entity   (BN + LSTM-GAN predictive models)      -- exists
    3. Bidirectional connection                                  -- THIS PACKAGE

This package provides the missing connection layer with three modules:

    state.py         TwinState per-entity tracking (observations, predictions,
                     feedback, drift EMA).

    uncertainty.py   Uncertainty quantification via noise-based MC sampling
                     and ensemble prediction for the LSTM-GAN generator.
                     Adapted for the stroke Generator architecture:
                     - 9 metadata features, 11 temporal features, 72 timesteps
                     - BCE-based training (not WGAN-GP)
                     - Plain Linear fc_in (no BatchNorm1d)

    drift.py         Drift detection via EMA thresholding and KS-test
                     distribution shift analysis.

Together these enable the question:
    "Given this patient's profile, what vital-sign trajectory do we expect,
     how confident are we, and has the model drifted from reality?"
"""

from src.connection.state import TwinState, PredictionRecord, FeedbackRecord
from src.connection.uncertainty import (
    mc_dropout_predict,
    ensemble_predict,
    prediction_interval,
    load_generator,
    Generator,
)
from src.connection.drift import compute_drift_report, detect_distribution_shift

__all__ = [
    "TwinState",
    "PredictionRecord",
    "FeedbackRecord",
    "mc_dropout_predict",
    "ensemble_predict",
    "prediction_interval",
    "load_generator",
    "Generator",
    "compute_drift_report",
    "detect_distribution_shift",
]
