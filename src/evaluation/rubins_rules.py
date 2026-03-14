"""Rubin's combining rules for pooling estimates from multiple synthetic datasets.

Implements the Reiter (2003) variant for fully synthetic data, where the
total variance accounts for both within-synthesis and between-synthesis
variability.

References
----------
- Reiter JP. Inference for partially synthetic, public use microdata sets.
  Survey Methodology. 2003;29(2):181-188.
- Raghunathan TE, Reiter JP, Rubin DB. Multiple imputation for statistical
  disclosure limitation. Journal of Official Statistics. 2003;19(4):1-16.
"""
import numpy as np
import scipy.stats as st


def pool_estimates(
    estimates: list[float],
    variances: list[float] | None = None,
) -> dict:
    """Pool point estimates from *m* synthetic datasets using Rubin's rules.

    Parameters
    ----------
    estimates : list[float]
        Point estimates Q_1, ..., Q_m from each synthetic dataset.
    variances : list[float] or None
        Within-synthesis variance estimates U_1, ..., U_m.  If ``None``,
        only the between-synthesis component is used (appropriate when
        within-dataset variance is not available, e.g. for single-number
        metrics like correlation or violation rate).

    Returns
    -------
    dict with keys:
        - ``pooled_estimate``: Q-bar (mean of estimates)
        - ``between_variance``: B (variance across synthetic datasets)
        - ``within_variance``: U-bar (mean of within-dataset variances, or 0)
        - ``total_variance``: T (Reiter 2003 formula)
        - ``ci_lower``, ``ci_upper``: 95% confidence interval
        - ``m``: number of synthetic datasets
    """
    m = len(estimates)
    q_bar = float(np.mean(estimates))

    # Between-synthesis variance
    b = float(np.var(estimates, ddof=1)) if m > 1 else 0.0

    # Within-synthesis variance
    if variances is not None:
        u_bar = float(np.mean(variances))
    else:
        u_bar = 0.0

    # Reiter (2003) total variance for fully synthetic data:
    #   T = U_bar / m  +  (1 + 1/m) * B
    # When within-variance is unavailable we fall back to just the
    # between-synthesis component.
    t = u_bar / m + (1 + 1 / m) * b

    # Degrees of freedom (Barnard-Rubin approximation)
    if t > 0 and b > 0:
        r = (1 + 1 / m) * b / t
        nu = (m - 1) * (1 + 1 / r) ** 2 if r > 0 else float("inf")
    else:
        nu = float("inf")

    # 95 % confidence interval
    if t > 0 and np.isfinite(nu) and nu > 0:
        t_crit = st.t.ppf(0.975, df=max(nu, 1))
        margin = t_crit * np.sqrt(t)
    else:
        margin = 1.96 * np.sqrt(t) if t > 0 else 0.0

    return {
        "pooled_estimate": q_bar,
        "between_variance": b,
        "within_variance": u_bar,
        "total_variance": t,
        "ci_lower": q_bar - margin,
        "ci_upper": q_bar + margin,
        "m": m,
    }


def pool_metric_dict(
    metric_dicts: list[dict],
) -> dict:
    """Pool a list of metric dictionaries (one per synthetic dataset).

    Each dictionary is expected to have the same keys mapping to scalar
    numeric values.  Returns a dictionary with the same keys, where each
    value is replaced by the pooled result from :func:`pool_estimates`.

    Example
    -------
    >>> dicts = [{"mean_dcr": 5.1, "mia_f1": 0.55},
    ...          {"mean_dcr": 5.3, "mia_f1": 0.53}]
    >>> pooled = pool_metric_dict(dicts)
    >>> pooled["mean_dcr"]["pooled_estimate"]
    5.2
    """
    if not metric_dicts:
        return {}

    keys = metric_dicts[0].keys()
    pooled: dict = {}
    for key in keys:
        values = []
        for d in metric_dicts:
            val = d.get(key)
            if isinstance(val, (int, float)) and np.isfinite(val):
                values.append(float(val))
        if values:
            pooled[key] = pool_estimates(values)
        else:
            pooled[key] = {"pooled_estimate": None, "ci_lower": None, "ci_upper": None}
    return pooled
