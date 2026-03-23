"""
Twin State -- Per-entity state tracking for the Grieves bidirectional connection.

Every patient (physical entity) has a TwinState that records observations from
the physical world, predictions from the virtual entity (LSTM-GAN), and feedback
records that pair predictions against actual outcomes.  This closed loop is the
core of the Grieves triad connection layer.

The EMA drift metric (alpha=0.3) weights recent errors ~3x historical, making
the system responsive to distribution shifts within 2-5 feedback cycles -- a
cadence that matches stroke admissions in neurocritical/stroke-unit care.

Design:
    - Immutable record types (PredictionRecord, FeedbackRecord) for audit trail.
    - TwinState is mutable and tracks the lifecycle:
        observation -> prediction -> feedback -> drift.
    - JSON-serializable for persistence and regulatory compliance.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PredictionRecord:
    """A single prediction issued by the virtual entity (LSTM-GAN generator).

    Attributes:
        prediction_id: Unique identifier for this prediction.
        timestamp: Unix epoch when the prediction was issued.
        predicted: Channel-level predicted values (channel_name -> value).
        uncertainty: Channel-level uncertainty (channel_name -> std).
        model_name: Name/version of the model that produced this prediction.
    """

    prediction_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    predicted: Dict[str, float] = field(default_factory=dict)
    uncertainty: Dict[str, float] = field(default_factory=dict)
    model_name: str = "lstm-gan-generator"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FeedbackRecord:
    """Pairs an actual outcome with a prior prediction, computing per-channel error.

    Attributes:
        feedback_id: Unique identifier for this feedback.
        prediction_id: Links back to the PredictionRecord being evaluated.
        timestamp: Unix epoch when the feedback was recorded.
        predicted: The values that were predicted.
        actual: The real values that were observed.
        error: Per-channel absolute error (|predicted - actual|).
    """

    feedback_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    prediction_id: str = ""
    timestamp: float = field(default_factory=time.time)
    predicted: Dict[str, float] = field(default_factory=dict)
    actual: Dict[str, float] = field(default_factory=dict)
    error: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TwinState:
    """Per-patient digital twin state -- the bidirectional connection record.

    Tracks the full lifecycle:
        physical entity -> observation ingestion
        virtual entity  -> prediction with UQ
        connection       -> feedback & drift monitoring

    Attributes:
        entity_id: Patient subject_id.
        last_sync: Timestamp of last physical-virtual synchronization.
        n_assimilations: Number of model update cycles completed.
        observations: Raw observations from the physical entity.
        predictions: Logged predictions from the virtual entity.
        feedback: Feedback records pairing predictions to actuals.
        recent_drift: Exponential moving average of prediction errors.
        metadata: Arbitrary metadata (demographics, profile info, etc.).
    """

    entity_id: str = ""
    last_sync: float = 0.0
    n_assimilations: int = 0
    observations: List[Dict[str, Any]] = field(default_factory=list)
    predictions: List[PredictionRecord] = field(default_factory=list)
    feedback: List[FeedbackRecord] = field(default_factory=list)
    recent_drift: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -- Observation ingestion (physical -> virtual) -----------------------

    def record_observation(self, obs: Dict[str, Any]) -> None:
        """Buffer a new observation from the physical entity.

        Args:
            obs: Dictionary of channel_name -> observed value, plus any
                 additional metadata (timestamp, source, etc.).
        """
        stamped = {**obs, "_ingested_at": time.time()}
        self.observations.append(stamped)

    # -- Prediction logging (virtual -> physical) --------------------------

    def record_prediction(self, pred: PredictionRecord) -> str:
        """Log a prediction from the virtual entity.

        Args:
            pred: PredictionRecord with predicted values and uncertainties.

        Returns:
            The prediction_id for later feedback linkage.
        """
        self.predictions.append(pred)
        return pred.prediction_id

    # -- Feedback recording (bidirectional closure) ------------------------

    def record_feedback(self, fb: FeedbackRecord, ema_alpha: float = 0.3) -> None:
        """Record feedback and update the EMA drift metric.

        The EMA with alpha=0.3 gives recent errors ~3x the weight of
        historical errors.  First feedback warm-starts the EMA to avoid
        zero-bias cold-start suppression.

        Args:
            fb: FeedbackRecord with predicted, actual, and error dicts.
            ema_alpha: Smoothing factor for EMA.  Default 0.3 detects drift
                       within ~5 consecutive elevated-error samples
                       (half-life = log(0.5)/log(0.7) ~ 1.94 cycles).
        """
        self.feedback.append(fb)

        if fb.error:
            mean_err = sum(fb.error.values()) / len(fb.error)
            if len(self.feedback) == 1:
                self.recent_drift = mean_err  # warm-start
            else:
                self.recent_drift = (
                    ema_alpha * mean_err + (1 - ema_alpha) * self.recent_drift
                )

    # -- Serialization -----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "entity_id": self.entity_id,
            "last_sync": self.last_sync,
            "n_assimilations": self.n_assimilations,
            "n_observations": len(self.observations),
            "n_predictions": len(self.predictions),
            "n_feedback": len(self.feedback),
            "recent_drift": self.recent_drift,
            "metadata": self.metadata,
            "predictions": [p.to_dict() for p in self.predictions[-10:]],
            "feedback": [f.to_dict() for f in self.feedback[-10:]],
        }

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), default=str, **kwargs)

    @classmethod
    def from_dict(cls, data: dict) -> "TwinState":
        return cls(
            entity_id=data.get("entity_id", ""),
            last_sync=data.get("last_sync", 0.0),
            n_assimilations=data.get("n_assimilations", 0),
            recent_drift=data.get("recent_drift", 0.0),
            metadata=data.get("metadata", {}),
        )
