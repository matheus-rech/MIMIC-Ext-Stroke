"""
Uncertainty Quantification -- Noise-based UQ for the LSTM-GAN generator.

The stroke digital twin's virtual entity is an LSTM-GAN generator that maps
(metadata, noise) -> temporal trajectory.  UQ exploits this stochastic
interface: same patient profile + different noise vectors = different
plausible trajectories.  The spread across draws gives epistemic + aleatoric
uncertainty estimates.

Three UQ strategies:

  1. mc_dropout_predict -- N forward passes with different noise draws.
     This is the primary method: the generator's noise input is the
     natural source of stochasticity.  No dropout needed because the
     noise *is* the latent perturbation.

  2. ensemble_predict -- If multiple generator checkpoints are available,
     average predictions across checkpoints for ensemble UQ.

  3. prediction_interval -- Gaussian (1-alpha) intervals from mean +/- z*std.

Architecture constants (must match src/models/dgan_model.py Generator):
    NOISE_DIM   = 100
    HIDDEN_DIM  = 128
    N_METADATA  = 9   (anchor_age, los, hospital_expire_flag, has_hypertension,
                        has_diabetes, has_afib, has_dyslipidemia, has_ckd, has_cad)
    N_FEATURES  = 11  (hr, sbp, dbp, map, rr, spo2, temp_c,
                        gcs_eye, gcs_verbal, gcs_motor, gcs_total)
    SEQ_LEN     = 72
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

# Constants matching the trained generator (src/models/dgan_model.py)
NOISE_DIM = 100
HIDDEN_DIM = 128
N_METADATA = 9
N_FEATURES = 11
SEQ_LEN = 72


class Generator(nn.Module):
    """LSTM-GAN generator (must match src/models/dgan_model.py architecture exactly).

    Input:  metadata (n, N_METADATA) + noise (n, NOISE_DIM)
    Output: temporal sequences (n, SEQ_LEN, N_FEATURES)

    Architecture:
        fc_in:  Linear(n_metadata + noise_dim, hidden_dim) -- no BatchNorm
        lstm:   LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True)
        fc_out: Linear(hidden_dim, n_features)
        tanh:   Tanh() activation on output

    Note: The stroke Generator differs from the TBI LSTMGenerator in that it
    uses a plain Linear layer for fc_in (no BatchNorm1d), uses hidden_dim for
    both LSTM input and hidden size, and applies ReLU via torch.relu() in
    forward() rather than nn.ReLU() in a Sequential block.
    """

    def __init__(
        self,
        n_metadata: int = N_METADATA,
        noise_dim: int = NOISE_DIM,
        hidden_dim: int = HIDDEN_DIM,
        n_features: int = N_FEATURES,
        seq_len: int = SEQ_LEN,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        # Embed metadata + noise
        self.fc_in = nn.Linear(n_metadata + noise_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, num_layers=2)
        self.fc_out = nn.Linear(hidden_dim, n_features)
        self.tanh = nn.Tanh()

    def forward(self, metadata: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # Combine metadata and noise
        x = torch.cat([metadata, noise], dim=-1)
        x = torch.relu(self.fc_in(x))
        # Repeat for sequence length
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)
        x, _ = self.lstm(x)
        x = self.fc_out(x)
        return self.tanh(x)  # Output in [-1, 1]


def load_generator(
    checkpoint_path: str = "./outputs/dgan/dgan_model.pt",
    device: Optional[torch.device] = None,
    n_metadata: int = N_METADATA,
    noise_dim: int = NOISE_DIM,
    hidden_dim: int = HIDDEN_DIM,
    n_features: int = N_FEATURES,
    seq_len: int = SEQ_LEN,
) -> Tuple[Generator, dict]:
    """Load generator weights from a StrokeTimeSeriesDGAN checkpoint.

    The stroke pipeline saves models via StrokeTimeSeriesDGAN.save(), which
    stores a dict with keys 'generator', 'discriminator', and 'config'.
    This function extracts the generator weights and config.

    Args:
        checkpoint_path: Path to the saved .pt checkpoint file.
        device: Torch device (defaults to CPU).
        n_metadata: Number of metadata features (default 9).
        noise_dim: Noise dimension (default 100).
        hidden_dim: Hidden dimension (default 128).
        n_features: Number of temporal features (default 11).
        seq_len: Sequence length (default 72).

    Returns:
        (generator, config) where config is the saved hyperparameter dict.
    """
    ckpt_path = Path(checkpoint_path)
    dev = device or torch.device("cpu")

    checkpoint = torch.load(str(ckpt_path), map_location=dev, weights_only=True)

    # If checkpoint contains 'config', use those values
    config = checkpoint.get("config", {})
    _n_metadata = config.get("n_metadata", n_metadata)
    _noise_dim = config.get("noise_dim", noise_dim)
    _hidden_dim = config.get("hidden_dim", hidden_dim)
    _n_features = config.get("n_features", n_features)
    _seq_len = config.get("seq_len", seq_len)

    gen = Generator(_n_metadata, _noise_dim, _hidden_dim, _n_features, _seq_len)

    # Load weights -- checkpoint may have 'generator' key or be raw state_dict
    if "generator" in checkpoint:
        gen.load_state_dict(checkpoint["generator"])
    else:
        gen.load_state_dict(checkpoint)

    gen.to(dev)
    # Set to inference mode
    gen.training = False

    logger.info(
        "Generator loaded: %d params, n_metadata=%d, n_features=%d, "
        "seq_len=%d, device=%s",
        sum(p.numel() for p in gen.parameters()),
        _n_metadata,
        _n_features,
        _seq_len,
        dev,
    )
    return gen, config


# ---------------------------------------------------------------------------
# Primary UQ: noise-based sampling
# ---------------------------------------------------------------------------

def mc_dropout_predict(
    generator: Generator,
    metadata: np.ndarray,
    n_samples: int = 30,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate n_samples trajectories with different noise, compute mean and std.

    This is noise-based UQ: same patient profile, different noise vectors
    produce different plausible trajectories.  The spread quantifies the
    generator's uncertainty about this patient's temporal evolution.

    Despite the name "mc_dropout", dropout is not used -- the generator's
    noise input *is* the stochastic element.  The name preserves the MC
    sampling interface expected by the connection layer.

    Args:
        generator: Trained Generator.
        metadata: Normalized static profile, shape (n_metadata,) or (1, n_metadata).
        n_samples: Number of noise draws (recommended 30-50).
        device: Torch device (defaults to CPU).

    Returns:
        mean_trajectory: shape (seq_len, n_features) = (72, 11).
        std_trajectory:  shape (seq_len, n_features) = (72, 11).
    """
    dev = device or torch.device("cpu")
    generator.to(dev)
    # Set to inference mode for deterministic BN/dropout behavior
    generator.training = False

    if metadata.ndim == 1:
        metadata = metadata[np.newaxis, :]

    meta_t = torch.tensor(
        np.tile(metadata, (n_samples, 1)),
        dtype=torch.float32,
        device=dev,
    )
    noise = torch.randn(n_samples, NOISE_DIM, device=dev)

    with torch.no_grad():
        trajectories = generator(meta_t, noise).cpu().numpy()  # (n_samples, 72, 11)

    mean_traj = np.mean(trajectories, axis=0)  # (72, 11)
    std_traj = np.std(trajectories, axis=0)    # (72, 11)

    logger.info(
        "MC noise-based UQ: %d samples, mean_std_range=[%.4f, %.4f]",
        n_samples,
        std_traj.min(),
        std_traj.max(),
    )
    return mean_traj, std_traj


def ensemble_predict(
    generators: List[Generator],
    metadata: np.ndarray,
    n_noise_per_model: int = 10,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Ensemble UQ from multiple generator checkpoints.

    If multiple generator checkpoints are available (e.g., from different
    training runs or epochs), ensemble them for improved uncertainty estimates.
    Each generator gets n_noise_per_model noise draws; the combined set
    captures both inter-model and intra-model variability.

    Args:
        generators: List of trained Generator instances.
        metadata: Normalized static profile, shape (n_metadata,) or (1, n_metadata).
        n_noise_per_model: Noise draws per generator.
        device: Torch device.

    Returns:
        mean_trajectory: shape (seq_len, n_features).
        std_trajectory:  shape (seq_len, n_features).
    """
    dev = device or torch.device("cpu")

    if metadata.ndim == 1:
        metadata = metadata[np.newaxis, :]

    all_trajectories = []

    for gen in generators:
        gen.to(dev)
        gen.training = False

        meta_t = torch.tensor(
            np.tile(metadata, (n_noise_per_model, 1)),
            dtype=torch.float32,
            device=dev,
        )
        noise = torch.randn(n_noise_per_model, NOISE_DIM, device=dev)

        with torch.no_grad():
            trajs = gen(meta_t, noise).cpu().numpy()
        all_trajectories.append(trajs)

    combined = np.concatenate(all_trajectories, axis=0)  # (total, 72, 11)
    mean_traj = np.mean(combined, axis=0)
    std_traj = np.std(combined, axis=0)

    logger.info(
        "Ensemble UQ: %d models x %d noise = %d total samples",
        len(generators),
        n_noise_per_model,
        combined.shape[0],
    )
    return mean_traj, std_traj


# ---------------------------------------------------------------------------
# Prediction intervals
# ---------------------------------------------------------------------------

def prediction_interval(
    mean: np.ndarray,
    std: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute (1-alpha) prediction interval from mean and std.

    Uses the Gaussian z-quantile for the interval.  For alpha=0.05, this
    gives a 95% prediction interval: mean +/- 1.96*std.

    Args:
        mean: Mean trajectory, shape (seq_len, n_features).
        std: Std trajectory, shape (seq_len, n_features).
        alpha: Significance level (0.05 = 95% interval).

    Returns:
        lower: Lower bound of interval, same shape as mean.
        upper: Upper bound of interval, same shape as mean.
    """
    z = sp_stats.norm.ppf(1 - alpha / 2)
    lower = mean - z * std
    upper = mean + z * std

    logger.info(
        "Prediction interval: alpha=%.3f, z=%.3f, coverage=%.1f%%",
        alpha,
        z,
        (1 - alpha) * 100,
    )
    return lower, upper
