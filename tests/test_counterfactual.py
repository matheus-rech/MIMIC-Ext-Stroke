# tests/test_counterfactual.py
import pandas as pd


def test_counterfactual_simulator_creates():
    from src.simulation.counterfactual import CounterfactualSimulator

    sim = CounterfactualSimulator()
    assert sim is not None


def test_simulate_scenario():
    from src.simulation.counterfactual import CounterfactualSimulator
    from src.models.hybrid import HybridDigitalTwin

    # Fit on tiny data
    static = pd.read_parquet("outputs/cohort/static_features.parquet").head(100)
    ts = pd.read_parquet("outputs/cohort/timeseries_processed.parquet")

    pipeline = HybridDigitalTwin(dgan_epochs=3, dgan_hidden_dim=16, dgan_noise_dim=8)
    pipeline.fit(static, ts)

    sim = CounterfactualSimulator(pipeline)

    # Get a patient profile
    patient = pipeline.generate_static(n=1, seed=42).iloc[0].to_dict()

    # Simulate: what if patient had afib?
    factual = sim.simulate_scenario(patient, intervention={})
    counterfactual = sim.simulate_scenario(patient, intervention={"has_afib": 1})

    assert factual["trajectories"].shape[1] > 0  # has timesteps
    assert counterfactual["trajectories"].shape[1] > 0


def test_compare_scenarios():
    from src.simulation.counterfactual import CounterfactualSimulator
    from src.models.hybrid import HybridDigitalTwin

    static = pd.read_parquet("outputs/cohort/static_features.parquet").head(100)
    ts = pd.read_parquet("outputs/cohort/timeseries_processed.parquet")

    pipeline = HybridDigitalTwin(dgan_epochs=3, dgan_hidden_dim=16, dgan_noise_dim=8)
    pipeline.fit(static, ts)

    sim = CounterfactualSimulator(pipeline)
    patient = pipeline.generate_static(n=1, seed=42).iloc[0].to_dict()

    scenarios = {
        "baseline": {},
        "add_afib": {"has_afib": 1},
        "younger": {"anchor_age": 45},
    }
    results = sim.compare_scenarios(patient, scenarios)
    assert "baseline" in results
    assert "add_afib" in results
    assert "younger" in results
