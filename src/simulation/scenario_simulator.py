"""Associational scenario simulation.

Given a synthetic patient profile, modify clinical attributes and regenerate
the time-series to explore how learned statistical associations translate into
different ICU trajectory patterns.

Important
---------
These simulations reflect **learned associations from observational data**,
not causal effects.  Results should be interpreted as hypothesis-generating
explorations, not evidence of causation.  See Section 2.10 of the manuscript
for a detailed discussion of the assumptions required for causal
interpretation and why they are unlikely to hold in this setting.
"""

import numpy as np
import pandas as pd


class ScenarioSimulator:
    """Simulate associational scenarios by modifying patient profiles."""

    def __init__(self, hybrid_model=None):
        self.model = hybrid_model

    def simulate_scenario(
        self,
        patient_profile: dict,
        modification: dict,
        n_samples: int = 10,
    ) -> dict:
        """Modify patient profile and regenerate trajectory.

        Parameters
        ----------
        patient_profile : dict
            A single patient's static features.
        modification : dict
            Attribute changes to apply,
            e.g. ``{"has_afib": 1, "anchor_age": 75}``.
        n_samples : int
            Number of trajectory samples to generate (for uncertainty).

        Returns
        -------
        dict with ``trajectories`` (ndarray, shape *n_samples x seq_len x
        n_features*) and ``modified_profile`` (dict).
        """
        modified = patient_profile.copy()
        modified.update(modification)

        profile_df = pd.DataFrame([modified])

        metadata = self.model._static_to_metadata(profile_df)
        metadata_repeated = np.repeat(metadata, n_samples, axis=0)
        trajectories = self.model.dgan.generate(metadata_repeated)

        return {
            "trajectories": trajectories,
            "modified_profile": modified,
            "n_samples": n_samples,
        }

    def compare_scenarios(
        self,
        patient_profile: dict,
        scenarios: dict,
        n_samples: int = 10,
    ) -> dict:
        """Compare multiple modification scenarios for one patient.

        Parameters
        ----------
        patient_profile : dict
            Baseline patient profile.
        scenarios : dict
            ``{name: modification_dict}`` mapping.

        Returns
        -------
        dict with results per scenario.
        """
        results = {}
        for name, modification in scenarios.items():
            result = self.simulate_scenario(patient_profile, modification, n_samples)
            trajs = result["trajectories"]
            results[name] = {
                "trajectories": trajs,
                "mean_trajectory": trajs.mean(axis=0),
                "std_trajectory": trajs.std(axis=0),
                "modified_profile": result["modified_profile"],
            }
        return results

    def associational_difference(
        self,
        patient_profile: dict,
        modification: dict,
        outcome_fn=None,
        n_samples: int = 50,
    ) -> dict:
        """Estimate the associational difference for a profile modification.

        This is **not** an individual treatment effect (ITE).  The difference
        reflects the change in predicted trajectories driven by learned
        statistical associations, which may be confounded.

        Parameters
        ----------
        patient_profile : dict
            Baseline patient profile.
        modification : dict
            The attribute changes to test.
        outcome_fn : callable, optional
            Function that takes a trajectories array and returns scalar
            outcomes per sample.  Default: mean of last-timestep values.
        n_samples : int
            Number of trajectory samples per arm.
        """
        if outcome_fn is None:

            def outcome_fn(trajs):
                return trajs[:, -1, :].mean(axis=1)

        baseline = self.simulate_scenario(patient_profile, {}, n_samples)
        modified = self.simulate_scenario(patient_profile, modification, n_samples)

        y_baseline = outcome_fn(baseline["trajectories"])
        y_modified = outcome_fn(modified["trajectories"])

        diff = y_modified.mean() - y_baseline.mean()

        return {
            "associational_difference": float(diff),
            "baseline_mean": float(y_baseline.mean()),
            "modified_mean": float(y_modified.mean()),
            "baseline_std": float(y_baseline.std()),
            "modified_std": float(y_modified.std()),
        }
