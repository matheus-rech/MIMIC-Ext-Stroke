"""Hybrid pipeline: BN (static profiles) + DGAN (ICU time-series).

This is the core digital twin generation pipeline. It:
1. Generates N static patient profiles via Bayesian Network
2. Prepares metadata features from those profiles for DGAN conditioning
3. Generates corresponding ICU time-series via DGAN conditioned on each profile
4. Combines static + temporal into complete synthetic patients
5. Repeats for >=10 independent datasets (El Emam 2024)
"""

import pandas as pd
import numpy as np

from src.models.bayesian_net import StrokeProfileBN
from src.models.dgan_model import StrokeTimeSeriesDGAN


class HybridDigitalTwin:
    """Hybrid digital twin: Bayesian Network for static profiles,
    DoppelGANger for ICU time-series, conditioned on static metadata.

    Parameters
    ----------
    bn_max_indegree : int
        Maximum number of parents per node in BN structure learning.
    dgan_epochs : int
        Number of DGAN training epochs.
    dgan_batch_size : int
        DGAN training batch size.
    dgan_noise_dim : int
        Dimension of noise vector for DGAN generator.
    dgan_hidden_dim : int
        Hidden dimension for DGAN LSTM layers.
    seq_len : int
        Length of generated time-series sequences (hours).
    """

    def __init__(
        self,
        bn_max_indegree: int = 3,
        dgan_epochs: int = 500,
        dgan_batch_size: int = 32,
        dgan_noise_dim: int = 100,
        dgan_hidden_dim: int = 128,
        seq_len: int = 72,
        loss_type: str = "bce",
        n_critic: int = 5,
        gp_lambda: float = 10.0,
    ):
        self.bn = StrokeProfileBN(max_indegree=bn_max_indegree)
        self.seq_len = seq_len
        self.dgan_params = {
            "epochs": dgan_epochs,
            "batch_size": dgan_batch_size,
            "noise_dim": dgan_noise_dim,
            "hidden_dim": dgan_hidden_dim,
            "loss_type": loss_type,
            "n_critic": n_critic,
            "gp_lambda": gp_lambda,
        }
        self.dgan: StrokeTimeSeriesDGAN | None = None
        self._metadata_cols: list[str] | None = None
        self._ts_feature_cols: list[str] | None = None
        self._metadata_stats: dict[str, np.ndarray] | None = None

    # ------------------------------------------------------------------ #
    #  Fitting                                                             #
    # ------------------------------------------------------------------ #

    def fit_static(self, static_df: pd.DataFrame) -> "HybridDigitalTwin":
        """Fit only the Bayesian Network on static features."""
        self.bn.fit(static_df)
        return self

    def fit(self, static_df: pd.DataFrame, timeseries_df: pd.DataFrame) -> "HybridDigitalTwin":
        """Fit both BN and DGAN models.

        1. Fit BN on static features
        2. Prepare metadata from static features for DGAN conditioning
        3. Reshape time-series to 3D tensor (patients x hours x features)
        4. Train DGAN on (metadata, time-series) pairs
        """
        # Fit BN
        self.fit_static(static_df)

        # Prepare DGAN training data
        metadata, sequences = self._prepare_dgan_data(static_df, timeseries_df)

        # Build and train DGAN
        n_features = sequences.shape[2]
        n_metadata = metadata.shape[1]
        self.dgan = StrokeTimeSeriesDGAN(
            n_features=n_features,
            n_metadata=n_metadata,
            seq_len=self.seq_len,
            **self.dgan_params,
        )
        self.dgan.train(metadata, sequences)
        return self

    # ------------------------------------------------------------------ #
    #  Generation                                                          #
    # ------------------------------------------------------------------ #

    def generate_static(self, n: int, seed: int = 42) -> pd.DataFrame:
        """Generate only static profiles via BN."""
        return self.bn.sample(n=n, seed=seed)

    def generate(self, n_patients: int, seed: int = 42) -> dict:
        """Generate complete synthetic patients (static + temporal).

        Returns
        -------
        dict with keys:
            'static' : pd.DataFrame of shape (n_patients, n_static_features)
            'timeseries' : np.ndarray of shape (n_patients, seq_len, n_ts_features)
        """
        if self.dgan is None:
            raise RuntimeError("DGAN not fitted. Call fit() with both static and timeseries data.")

        # Generate static profiles
        static = self.bn.sample(n=n_patients, seed=seed)

        # Prepare metadata for DGAN conditioning
        metadata = self._static_to_metadata(static)

        # Generate time-series conditioned on metadata
        timeseries = self.dgan.generate(metadata)

        return {"static": static, "timeseries": timeseries}

    def generate_multiple_datasets(
        self,
        n_patients: int,
        n_datasets: int = 10,
        base_seed: int = 42,
    ) -> list[dict]:
        """Generate multiple independent synthetic datasets (El Emam 2024).

        Each dataset uses a different seed for both BN sampling and DGAN
        generation, ensuring independence across datasets.

        Parameters
        ----------
        n_patients : int
            Number of patients per dataset.
        n_datasets : int
            Number of independent datasets to generate (>=10 recommended).
        base_seed : int
            Base seed; each dataset uses base_seed + i.

        Returns
        -------
        list of dicts, each with 'static' and 'timeseries' keys.
        """
        datasets = []
        for i in range(n_datasets):
            seed = base_seed + i
            result = self.generate(n_patients=n_patients, seed=seed)
            datasets.append(result)
        return datasets

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _prepare_dgan_data(
        self, static_df: pd.DataFrame, ts_df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert DataFrames to numpy arrays for DGAN.

        - Metadata: select numeric features from static that serve as conditioning
        - Sequences: reshape long-format TS to 3D tensor, pad/truncate to seq_len
        """
        # Select metadata columns (numeric features for conditioning)
        metadata_candidates = [
            "anchor_age",
            "los",
            "hospital_expire_flag",
            "has_hypertension",
            "has_diabetes",
            "has_afib",
            "has_dyslipidemia",
            "has_ckd",
            "has_cad",
        ]
        self._metadata_cols = [c for c in metadata_candidates if c in static_df.columns]

        # Time-series feature columns
        ts_feature_candidates = [
            "hr",
            "sbp",
            "dbp",
            "map",
            "rr",
            "spo2",
            "temp_c",
            "gcs_eye",
            "gcs_verbal",
            "gcs_motor",
            "gcs_total",
        ]
        self._ts_feature_cols = [c for c in ts_feature_candidates if c in ts_df.columns]

        # Get patients that have both static and time-series data
        ts_patients = set(ts_df["stay_id"].unique())
        static_with_ts = static_df[static_df["stay_id"].isin(ts_patients)].copy()

        if len(static_with_ts) == 0:
            raise ValueError(
                "No overlapping patients between static and timeseries data. "
                "Check that both DataFrames share 'stay_id' values."
            )

        # Normalize metadata
        meta_values = static_with_ts[self._metadata_cols].values.astype(np.float32)
        self._metadata_stats = {
            "mean": np.nanmean(meta_values, axis=0),
            "std": np.nanstd(meta_values, axis=0) + 1e-8,
        }
        meta_norm = (meta_values - self._metadata_stats["mean"]) / self._metadata_stats["std"]
        meta_norm = np.nan_to_num(meta_norm, nan=0.0)

        # Reshape time-series to 3D: (n_patients, seq_len, n_features)
        sequences = []
        valid_indices = []
        for idx, (_, row) in enumerate(static_with_ts.iterrows()):
            patient_ts = ts_df[ts_df["stay_id"] == row["stay_id"]].sort_values("hour")
            ts_matrix = patient_ts[self._ts_feature_cols].values.astype(np.float32)

            if len(ts_matrix) == 0:
                continue

            # Pad or truncate to seq_len
            if len(ts_matrix) >= self.seq_len:
                ts_matrix = ts_matrix[: self.seq_len]
            else:
                pad = np.zeros(
                    (self.seq_len - len(ts_matrix), len(self._ts_feature_cols)),
                    dtype=np.float32,
                )
                ts_matrix = np.vstack([ts_matrix, pad])

            # Replace NaN with 0 for training
            ts_matrix = np.nan_to_num(ts_matrix, nan=0.0)

            sequences.append(ts_matrix)
            valid_indices.append(idx)

        metadata_out = meta_norm[valid_indices]
        sequences_out = np.array(sequences)

        return metadata_out, sequences_out

    def _static_to_metadata(self, static_df: pd.DataFrame) -> np.ndarray:
        """Convert generated static profiles to normalized metadata array for DGAN.

        Uses the same normalization stats computed during fit.
        """
        if self._metadata_cols is None or self._metadata_stats is None:
            raise RuntimeError("Metadata columns/stats not initialized. Call fit() first.")

        meta_values = static_df[self._metadata_cols].values.astype(np.float32)
        meta_norm = (meta_values - self._metadata_stats["mean"]) / self._metadata_stats["std"]
        return np.nan_to_num(meta_norm, nan=0.0)
