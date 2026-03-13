"""CTGAN and TVAE baselines for comparison with Bayesian Network approach.

Uses the SDV library's single-table synthesizers.
"""

import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer


class StrokeCTGAN:
    """CTGAN wrapper for synthetic static profile generation."""

    def __init__(self, epochs: int = 300, batch_size: int = 500):
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self._metadata = None

    def fit(self, df: pd.DataFrame) -> "StrokeCTGAN":
        """Fit CTGAN on static features."""
        self._metadata = SingleTableMetadata()
        self._metadata.detect_from_dataframe(df)

        effective_batch = min(self.batch_size, len(df))
        self.model = CTGANSynthesizer(
            self._metadata,
            epochs=self.epochs,
            batch_size=effective_batch,
            verbose=False,
        )
        self.model.fit(df)
        return self

    def sample(self, n: int) -> pd.DataFrame:
        """Generate n synthetic profiles."""
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.model.sample(num_rows=n)

    def save(self, path: str) -> None:
        """Persist fitted model to disk."""
        self.model.save(path)

    @classmethod
    def load(cls, path: str) -> "StrokeCTGAN":
        """Load a previously saved model."""
        obj = cls()
        obj.model = CTGANSynthesizer.load(path)
        return obj


class StrokeTVAE:
    """TVAE wrapper for comparison."""

    def __init__(self, epochs: int = 300, batch_size: int = 500):
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None

    def fit(self, df: pd.DataFrame) -> "StrokeTVAE":
        """Fit TVAE on static features."""
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df)

        effective_batch = min(self.batch_size, len(df))
        self.model = TVAESynthesizer(
            metadata,
            epochs=self.epochs,
            batch_size=effective_batch,
        )
        self.model.fit(df)
        return self

    def sample(self, n: int) -> pd.DataFrame:
        """Generate n synthetic profiles."""
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.model.sample(num_rows=n)

    def save(self, path: str) -> None:
        """Persist fitted model to disk."""
        self.model.save(path)

    @classmethod
    def load(cls, path: str) -> "StrokeTVAE":
        """Load a previously saved model."""
        obj = cls()
        obj.model = TVAESynthesizer.load(path)
        return obj
