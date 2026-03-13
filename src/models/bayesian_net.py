"""Bayesian Network for static stroke patient profile generation."""

import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import HillClimbSearch, BIC, BayesianEstimator
from pgmpy.factors.discrete import TabularCPD
# BayesianModelSampling not needed; using DiscreteBayesianNetwork.simulate()


class StrokeProfileBN:
    """Learn and sample from a Bayesian Network over static stroke profiles.

    Selects clinically meaningful features, discretizes continuous variables,
    learns DAG structure via Hill Climbing with BIC scoring, fits parameters
    with Bayesian estimation (BDeu prior), and generates synthetic profiles
    via forward sampling.
    """

    BN_FEATURES = [
        "anchor_age",
        "gender",
        "stroke_subtype",
        "hospital_expire_flag",
        "los",
        "has_hypertension",
        "has_diabetes",
        "has_afib",
        "has_dyslipidemia",
        "has_ckd",
        "has_cad",
        "lab_glucose",
        "lab_sodium",
        "lab_creatinine",
        "lab_hemoglobin",
        "lab_platelets",
        "lab_inr",
    ]

    # Discretization bins for continuous variables
    _AGE_BINS = [0, 45, 55, 65, 75, 85, 200]
    _AGE_LABELS = ["18-45", "45-55", "55-65", "65-75", "75-85", "85+"]

    _LOS_BINS = [0, 1, 3, 7, 14, 1e6]
    _LOS_LABELS = ["0-1d", "1-3d", "3-7d", "7-14d", "14+d"]

    # Lab discretization: quartile-based (computed from training data)
    _LAB_COLS = [
        "lab_glucose",
        "lab_sodium",
        "lab_creatinine",
        "lab_hemoglobin",
        "lab_platelets",
        "lab_inr",
    ]

    def __init__(self, max_indegree: int = 3, scoring: str = "bic"):
        self.max_indegree = max_indegree
        self.scoring = scoring
        self.model = None
        self._feature_cols: list[str] | None = None
        self._lab_bin_edges: dict[str, np.ndarray] = {}
        self._lab_labels: dict[str, list[str]] = {}

    def fit(self, df: pd.DataFrame) -> "StrokeProfileBN":
        """Learn BN structure and parameters from static features."""
        available = [c for c in self.BN_FEATURES if c in df.columns]
        self._feature_cols = available
        data = self._discretize(df[available].copy())

        # Structure learning
        hc = HillClimbSearch(data)
        best_dag = hc.estimate(
            scoring_method=BIC(data),
            max_indegree=self.max_indegree,
        )

        self.model = DiscreteBayesianNetwork(best_dag.edges())

        # Ensure all columns appear as nodes (including isolated ones)
        for col in data.columns:
            if col not in self.model.nodes():
                self.model.add_node(col)

        # Parameter learning with Bayesian estimation + BDeu smoothing
        self.model.fit(
            data,
            estimator=BayesianEstimator,
            prior_type="BDeu",
            equivalent_sample_size=10,
        )

        # Fix: pgmpy's fit() skips isolated nodes — add marginal CPDs manually
        for col in data.columns:
            if self.model.get_cpds(col) is None:
                states = sorted(data[col].unique())
                counts = data[col].value_counts()
                total = counts.sum()
                probs = [[counts.get(s, 0) / total] for s in states]
                cpd = TabularCPD(
                    variable=col,
                    variable_card=len(states),
                    values=probs,
                    state_names={col: states},
                )
                self.model.add_cpds(cpd)

        return self

    def sample(self, n: int, seed: int = 42) -> pd.DataFrame:
        """Generate n synthetic patient profiles."""
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        synthetic = self.model.simulate(
            n_samples=n, seed=seed, show_progress=False
        )

        return self._inverse_discretize(synthetic)

    def get_dag(self) -> list[tuple[str, str]]:
        """Return learned DAG edges."""
        if self.model is None:
            return []
        return list(self.model.edges())

    # ------------------------------------------------------------------ #
    #  Discretization helpers                                              #
    # ------------------------------------------------------------------ #

    def _discretize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Discretize continuous variables for BN structure learning."""
        result = df.copy()

        # --- Age ---
        if "anchor_age" in result.columns:
            result["anchor_age"] = pd.cut(
                result["anchor_age"],
                bins=self._AGE_BINS,
                labels=self._AGE_LABELS,
                right=False,
            ).astype(str)

        # --- Length of stay ---
        if "los" in result.columns:
            result["los"] = pd.cut(
                result["los"].clip(lower=0),
                bins=self._LOS_BINS,
                labels=self._LOS_LABELS,
                right=False,
            ).astype(str)

        # --- Labs: quartile-based discretization ---
        for col in self._LAB_COLS:
            if col not in result.columns:
                continue

            # Impute NaN with median before discretization
            median_val = result[col].median()
            filled = result[col].fillna(median_val)

            # Compute quartile edges from training data
            try:
                _, edges = pd.qcut(filled, q=4, retbins=True, duplicates="drop")
            except ValueError:
                # Fallback: equal-width bins if quartiles fail
                _, edges = pd.cut(filled, bins=4, retbins=True, duplicates="drop")

            self._lab_bin_edges[col] = edges
            labels = [f"Q{i+1}" for i in range(len(edges) - 1)]
            self._lab_labels[col] = labels

            result[col] = pd.cut(
                filled, bins=edges, labels=labels, include_lowest=True
            ).astype(str)

        # --- hospital_expire_flag: already 0/1 int, convert to str ---
        if "hospital_expire_flag" in result.columns:
            result["hospital_expire_flag"] = result["hospital_expire_flag"].astype(str)

        # --- Binary comorbidities: convert to str ---
        binary_cols = [
            "has_hypertension",
            "has_diabetes",
            "has_afib",
            "has_dyslipidemia",
            "has_ckd",
            "has_cad",
        ]
        for col in binary_cols:
            if col in result.columns:
                result[col] = result[col].astype(str)

        # --- Categorical columns already string: gender, stroke_subtype ---
        # Ensure everything is string type for pgmpy
        for col in result.columns:
            result[col] = result[col].astype(str)

        return result

    def _inverse_discretize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert discretized samples back to mixed types.

        - Continuous bins → midpoint + uniform noise within the bin
        - Binary flags → int
        - Categoricals → kept as-is
        """
        rng = np.random.default_rng(0)
        result = df.copy()

        # --- Age ---
        if "anchor_age" in result.columns:
            age_map = {
                "18-45": (18, 45),
                "45-55": (45, 55),
                "55-65": (55, 65),
                "65-75": (65, 75),
                "75-85": (75, 85),
                "85+": (85, 100),
            }
            result["anchor_age"] = result["anchor_age"].map(
                lambda x: self._midpoint_noise(age_map.get(x, (65, 75)), rng)
            )

        # --- LOS ---
        if "los" in result.columns:
            los_map = {
                "0-1d": (0, 1),
                "1-3d": (1, 3),
                "3-7d": (3, 7),
                "7-14d": (7, 14),
                "14+d": (14, 30),
            }
            result["los"] = result["los"].map(
                lambda x: round(self._midpoint_noise(los_map.get(x, (3, 7)), rng), 2)
            )

        # --- Labs ---
        for col in self._LAB_COLS:
            if col not in result.columns or col not in self._lab_bin_edges:
                continue
            edges = self._lab_bin_edges[col]
            labels = self._lab_labels[col]
            bin_ranges = {
                labels[i]: (edges[i], edges[i + 1]) for i in range(len(labels))
            }
            result[col] = result[col].map(
                lambda x, br=bin_ranges: self._midpoint_noise(
                    br.get(x, (edges[0], edges[-1])), rng
                )
            )

        # --- Binary flags back to int ---
        binary_cols = [
            "hospital_expire_flag",
            "has_hypertension",
            "has_diabetes",
            "has_afib",
            "has_dyslipidemia",
            "has_ckd",
            "has_cad",
        ]
        for col in binary_cols:
            if col in result.columns:
                result[col] = result[col].astype(int)

        return result

    @staticmethod
    def _midpoint_noise(bounds: tuple[float, float], rng) -> float:
        """Return a random value within the bin range."""
        lo, hi = bounds
        return rng.uniform(lo, hi)
