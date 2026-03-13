"""Tests for DoppelGANger-style ICU time-series generation model."""
import pytest
import torch
import numpy as np


def test_dgan_builds():
    from src.models.dgan_model import StrokeTimeSeriesDGAN
    model = StrokeTimeSeriesDGAN(
        n_features=10, n_metadata=5, seq_len=72,
        noise_dim=16, hidden_dim=32, epochs=2, batch_size=4
    )
    assert model is not None


def test_dgan_trains_on_tiny_data():
    from src.models.dgan_model import StrokeTimeSeriesDGAN
    # Tiny synthetic data: 20 patients, 24 timesteps, 5 features
    n, t, f = 20, 24, 5
    metadata = np.random.randn(n, 3).astype(np.float32)
    sequences = np.random.randn(n, t, f).astype(np.float32)

    model = StrokeTimeSeriesDGAN(
        n_features=f, n_metadata=3, seq_len=t,
        noise_dim=8, hidden_dim=16, epochs=5, batch_size=4
    )
    model.train(metadata, sequences)
    # Should complete without error


def test_dgan_generates_correct_shape():
    from src.models.dgan_model import StrokeTimeSeriesDGAN
    n, t, f = 20, 24, 5
    metadata = np.random.randn(n, 3).astype(np.float32)
    sequences = np.random.randn(n, t, f).astype(np.float32)

    model = StrokeTimeSeriesDGAN(
        n_features=f, n_metadata=3, seq_len=t,
        noise_dim=8, hidden_dim=16, epochs=5, batch_size=4
    )
    model.train(metadata, sequences)

    # Generate conditioned on new metadata
    new_meta = np.random.randn(10, 3).astype(np.float32)
    generated = model.generate(new_meta)
    assert generated.shape == (10, t, f)


def test_dgan_output_bounded():
    """Generated values should be in [-1, 1] due to tanh activation."""
    from src.models.dgan_model import StrokeTimeSeriesDGAN
    n, t, f = 20, 24, 5
    metadata = np.random.randn(n, 3).astype(np.float32)
    sequences = np.random.randn(n, t, f).astype(np.float32)

    model = StrokeTimeSeriesDGAN(
        n_features=f, n_metadata=3, seq_len=t,
        noise_dim=8, hidden_dim=16, epochs=2, batch_size=4
    )
    model.train(metadata, sequences)

    new_meta = np.random.randn(5, 3).astype(np.float32)
    generated = model.generate(new_meta)
    assert np.all(generated >= -1.0) and np.all(generated <= 1.0)


def test_dgan_losses_recorded():
    """Training should record loss history."""
    from src.models.dgan_model import StrokeTimeSeriesDGAN
    n, t, f = 20, 24, 5
    metadata = np.random.randn(n, 3).astype(np.float32)
    sequences = np.random.randn(n, t, f).astype(np.float32)

    model = StrokeTimeSeriesDGAN(
        n_features=f, n_metadata=3, seq_len=t,
        noise_dim=8, hidden_dim=16, epochs=5, batch_size=4
    )
    model.train(metadata, sequences)

    assert len(model.losses["g_loss"]) == 5
    assert len(model.losses["d_loss"]) == 5


def test_dgan_save_load(tmp_path):
    """Model should save and load correctly."""
    from src.models.dgan_model import StrokeTimeSeriesDGAN
    n, t, f = 20, 24, 5
    metadata = np.random.randn(n, 3).astype(np.float32)
    sequences = np.random.randn(n, t, f).astype(np.float32)

    model = StrokeTimeSeriesDGAN(
        n_features=f, n_metadata=3, seq_len=t,
        noise_dim=8, hidden_dim=16, epochs=2, batch_size=4
    )
    model.train(metadata, sequences)

    save_path = tmp_path / "dgan_test.pt"
    model.save(str(save_path))
    assert save_path.exists()

    model2 = StrokeTimeSeriesDGAN(
        n_features=f, n_metadata=3, seq_len=t,
        noise_dim=8, hidden_dim=16, epochs=2, batch_size=4
    )
    model2.load(str(save_path))

    # Both models should generate same output given same noise seed
    test_meta = np.random.randn(3, 3).astype(np.float32)
    torch.manual_seed(42)
    out1 = model.generate(test_meta)
    torch.manual_seed(42)
    out2 = model2.generate(test_meta)
    np.testing.assert_array_almost_equal(out1, out2, decimal=5)


def test_dgan_generate_multiple_per_patient():
    """Generate multiple sequences per patient."""
    from src.models.dgan_model import StrokeTimeSeriesDGAN
    n, t, f = 20, 24, 5
    metadata = np.random.randn(n, 3).astype(np.float32)
    sequences = np.random.randn(n, t, f).astype(np.float32)

    model = StrokeTimeSeriesDGAN(
        n_features=f, n_metadata=3, seq_len=t,
        noise_dim=8, hidden_dim=16, epochs=2, batch_size=4
    )
    model.train(metadata, sequences)

    new_meta = np.random.randn(5, 3).astype(np.float32)
    generated = model.generate(new_meta, n_per_patient=3)
    assert generated.shape == (15, t, f)
