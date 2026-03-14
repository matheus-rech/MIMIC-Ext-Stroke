"""DoppelGANger-inspired model for ICU time-series generation.

Custom PyTorch implementation since gretel-synthetics is not installed.
Generates ICU vital sign trajectories conditioned on static patient metadata.

Reference: Lin et al., "Using GANs for Sharing Networked Time Series Data:
Challenges, Initial Promise, and Open Questions" (IMC 2020).
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


class Generator(nn.Module):
    """LSTM-based generator: metadata + noise -> time-series."""

    def __init__(
        self, n_metadata: int, noise_dim: int, hidden_dim: int, n_features: int, seq_len: int
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


class Discriminator(nn.Module):
    """Discriminator: (metadata, time-series) -> real/fake."""

    def __init__(self, n_metadata: int, n_features: int, hidden_dim: int, seq_len: int):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_dim, batch_first=True, num_layers=2)
        self.fc_meta = nn.Linear(n_metadata, hidden_dim)
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, metadata: torch.Tensor, sequence: torch.Tensor) -> torch.Tensor:
        # Process sequence with LSTM
        _, (h_n, _) = self.lstm(sequence)
        seq_embed = h_n[-1]  # Last layer hidden state
        # Process metadata
        meta_embed = torch.relu(self.fc_meta(metadata))
        # Combine and classify
        combined = torch.cat([seq_embed, meta_embed], dim=-1)
        return self.fc_out(combined)


class StrokeTimeSeriesDGAN:
    """DoppelGANger-style model for conditional ICU time-series generation.

    Trains a GAN with an LSTM-based generator and discriminator to produce
    realistic ICU vital sign trajectories conditioned on static patient
    metadata (age, comorbidities, stroke severity, etc.).

    Parameters
    ----------
    n_features : int
        Number of time-series features (e.g., vital signs).
    n_metadata : int
        Number of static metadata features.
    seq_len : int
        Length of generated sequences (number of timesteps).
    noise_dim : int
        Dimension of the noise vector for the generator.
    hidden_dim : int
        Hidden dimension for LSTM layers and feed-forward layers.
    epochs : int
        Number of training epochs.
    batch_size : int
        Training batch size.
    lr : float
        Learning rate for Adam optimizer.
    """

    def __init__(
        self,
        n_features: int,
        n_metadata: int,
        seq_len: int,
        noise_dim: int = 100,
        hidden_dim: int = 128,
        epochs: int = 500,
        batch_size: int = 32,
        lr: float = 0.0002,
    ):
        self.n_features = n_features
        self.n_metadata = n_metadata
        self.seq_len = seq_len
        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

        # Use MPS (Apple Silicon) if available, else CPU
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.generator = Generator(n_metadata, noise_dim, hidden_dim, n_features, seq_len).to(
            self.device
        )
        self.discriminator = Discriminator(n_metadata, n_features, hidden_dim, seq_len).to(
            self.device
        )

        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999)
        )

        self.losses: dict[str, list[float]] = {"g_loss": [], "d_loss": []}

    def train(self, metadata: np.ndarray, sequences: np.ndarray) -> None:
        """Train the DGAN model.

        Parameters
        ----------
        metadata : np.ndarray, shape (n_patients, n_metadata)
            Static patient features.
        sequences : np.ndarray, shape (n_patients, seq_len, n_features)
            Time-series data for each patient.
        """
        self.generator.train()
        self.discriminator.train()

        meta_tensor = torch.FloatTensor(metadata).to(self.device)
        seq_tensor = torch.FloatTensor(sequences).to(self.device)
        dataset = TensorDataset(meta_tensor, seq_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        criterion = nn.BCEWithLogitsLoss()

        for epoch in range(self.epochs):
            g_losses, d_losses = [], []
            for meta_batch, seq_batch in loader:
                batch_size = meta_batch.size(0)
                real_labels = torch.ones(batch_size, 1, device=self.device)
                fake_labels = torch.zeros(batch_size, 1, device=self.device)

                # ---------------------
                # Train Discriminator
                # ---------------------
                self.d_optimizer.zero_grad()
                d_real = self.discriminator(meta_batch, seq_batch)
                d_loss_real = criterion(d_real, real_labels)

                noise = torch.randn(batch_size, self.noise_dim, device=self.device)
                fake_seq = self.generator(meta_batch, noise)
                d_fake = self.discriminator(meta_batch, fake_seq.detach())
                d_loss_fake = criterion(d_fake, fake_labels)

                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                self.d_optimizer.step()

                # ---------------------
                # Train Generator
                # ---------------------
                self.g_optimizer.zero_grad()
                noise = torch.randn(batch_size, self.noise_dim, device=self.device)
                fake_seq = self.generator(meta_batch, noise)
                d_fake = self.discriminator(meta_batch, fake_seq)
                g_loss = criterion(d_fake, real_labels)
                g_loss.backward()
                self.g_optimizer.step()

                g_losses.append(g_loss.item())
                d_losses.append(d_loss.item())

            self.losses["g_loss"].append(float(np.mean(g_losses)))
            self.losses["d_loss"].append(float(np.mean(d_losses)))

    def generate(self, metadata: np.ndarray, n_per_patient: int = 1) -> np.ndarray:
        """Generate time-series conditioned on metadata.

        Parameters
        ----------
        metadata : np.ndarray, shape (n_patients, n_metadata)
            Static features for conditioning.
        n_per_patient : int
            Number of sequences to generate per patient.

        Returns
        -------
        np.ndarray, shape (n_patients * n_per_patient, seq_len, n_features)
        """
        self.generator.eval()
        with torch.no_grad():
            meta = torch.FloatTensor(metadata).to(self.device)
            if n_per_patient > 1:
                meta = meta.repeat(n_per_patient, 1)
            noise = torch.randn(meta.size(0), self.noise_dim, device=self.device)
            generated = self.generator(meta, noise)
        self.generator.train()
        return generated.cpu().numpy()

    def save(self, path: str) -> None:
        """Save model state to disk."""
        torch.save(
            {
                "generator": self.generator.state_dict(),
                "discriminator": self.discriminator.state_dict(),
                "config": {
                    "n_features": self.n_features,
                    "n_metadata": self.n_metadata,
                    "seq_len": self.seq_len,
                    "noise_dim": self.noise_dim,
                    "hidden_dim": self.hidden_dim,
                },
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load model state from disk."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.generator.load_state_dict(checkpoint["generator"])
        self.discriminator.load_state_dict(checkpoint["discriminator"])

    def parameter_count(self) -> dict[str, int]:
        """Return parameter counts for generator and discriminator."""
        g_params = sum(p.numel() for p in self.generator.parameters())
        d_params = sum(p.numel() for p in self.discriminator.parameters())
        return {"generator": g_params, "discriminator": d_params, "total": g_params + d_params}
