"""Counterfactual simulation — the 'digital twin' capability.

Given a synthetic patient profile, modify treatment/intervention variables
and regenerate the time-series to simulate alternative scenarios.
This is what distinguishes a digital twin from mere synthetic data.
"""
import numpy as np
import pandas as pd


class CounterfactualSimulator:
    """Simulate counterfactual scenarios by modifying patient profiles."""

    def __init__(self, hybrid_model=None):
        self.model = hybrid_model

    def simulate_scenario(self, patient_profile: dict, intervention: dict,
                          n_samples: int = 10) -> dict:
        """Modify patient profile and regenerate trajectory.

        Parameters
        ----------
        patient_profile : dict
            A single patient's static features
        intervention : dict
            Modifications to apply, e.g. {"has_afib": 1, "anchor_age": 75}
        n_samples : int
            Number of trajectory samples to generate (for uncertainty)

        Returns
        -------
        dict with 'trajectories' (np.ndarray shape n_samples x seq_len x n_features)
        and 'modified_profile' (dict)
        """
        modified = patient_profile.copy()
        modified.update(intervention)

        # Create DataFrame for the model
        profile_df = pd.DataFrame([modified])

        # Generate metadata and trajectories
        metadata = self.model._static_to_metadata(profile_df)
        # Repeat metadata for n_samples
        metadata_repeated = np.repeat(metadata, n_samples, axis=0)
        trajectories = self.model.dgan.generate(metadata_repeated)

        return {
            "trajectories": trajectories,
            "modified_profile": modified,
            "n_samples": n_samples,
        }

    def compare_scenarios(self, patient_profile: dict, scenarios: dict,
                          n_samples: int = 10) -> dict:
        """Compare multiple intervention scenarios for one patient.

        Parameters
        ----------
        patient_profile : dict
            Baseline patient profile
        scenarios : dict
            {name: intervention_dict} mapping

        Returns
        -------
        dict with results per scenario
        """
        results = {}
        for name, intervention in scenarios.items():
            result = self.simulate_scenario(patient_profile, intervention, n_samples)

            # Compute summary statistics per trajectory
            trajs = result["trajectories"]
            results[name] = {
                "trajectories": trajs,
                "mean_trajectory": trajs.mean(axis=0),
                "std_trajectory": trajs.std(axis=0),
                "modified_profile": result["modified_profile"],
            }

        return results

    def treatment_effect(self, patient_profile: dict,
                         treatment: dict, outcome_fn=None,
                         n_samples: int = 50) -> dict:
        """Estimate individual treatment effect.

        Parameters
        ----------
        treatment : dict
            The intervention to test
        outcome_fn : callable
            Function that takes trajectories array and returns scalar outcomes.
            Default: mean of last timestep values.
        """
        if outcome_fn is None:
            def outcome_fn(trajs):
                return trajs[:, -1, :].mean(axis=1)

        factual = self.simulate_scenario(patient_profile, {}, n_samples)
        counterfactual = self.simulate_scenario(patient_profile, treatment, n_samples)

        y0 = outcome_fn(factual["trajectories"])
        y1 = outcome_fn(counterfactual["trajectories"])

        ite = y1.mean() - y0.mean()

        return {
            "ite": float(ite),
            "y0_mean": float(y0.mean()),
            "y1_mean": float(y1.mean()),
            "y0_std": float(y0.std()),
            "y1_std": float(y1.std()),
        }
