"""Backward-compatibility shim — use :mod:`scenario_simulator` instead.

.. deprecated::
    This module used causal language ("counterfactual", "treatment effect")
    that overstated the capabilities of observational Bayesian Networks.
    All functionality has been moved to
    :class:`~src.simulation.scenario_simulator.ScenarioSimulator` with
    corrected associational terminology.
"""

import warnings

from src.simulation.scenario_simulator import ScenarioSimulator


class CounterfactualSimulator(ScenarioSimulator):
    """Deprecated — use :class:`ScenarioSimulator` instead."""

    def __init__(self, hybrid_model=None):
        warnings.warn(
            "CounterfactualSimulator is deprecated. "
            "Use ScenarioSimulator from src.simulation.scenario_simulator instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(hybrid_model=hybrid_model)

    def simulate_scenario(
        self, patient_profile: dict, intervention: dict, n_samples: int = 10
    ) -> dict:
        """Accept legacy *intervention* keyword and forward to *modification*."""
        return super().simulate_scenario(patient_profile, intervention, n_samples)

    def treatment_effect(
        self, patient_profile: dict, treatment: dict, outcome_fn=None, n_samples: int = 50
    ) -> dict:
        """Deprecated — use :meth:`associational_difference` instead."""
        warnings.warn(
            "treatment_effect() is deprecated. Use associational_difference() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        result = self.associational_difference(patient_profile, treatment, outcome_fn, n_samples)
        # Map new keys back to legacy keys for backward compatibility
        return {
            "ite": result["associational_difference"],
            "y0_mean": result["baseline_mean"],
            "y1_mean": result["modified_mean"],
            "y0_std": result["baseline_std"],
            "y1_std": result["modified_std"],
        }
