r"""Honest statistical inference for time-dependent forecasts.

Closes audit critical C-22. The asymptotic Wilson CIs that the paper
addendum used to bracket pooled out-of-fold accuracy assume i.i.d.
Bernoulli outcomes. Out-of-fold predictions on financial time series
are *not* i.i.d.: at horizon ``h>1`` they are explicitly overlapping,
and even at ``h=1`` they exhibit the autocorrelation structure of
daily returns. This module provides leakage-aware replacements:

- :func:`stationary_block_bootstrap` — Politis-Romano (1994) with
  geometric block lengths. Robust to unknown serial dependence.
- :func:`block_bootstrap_metric` — generic wrapper for any scalar
  metric on a time-aligned array.
- :func:`block_bootstrap_accuracy` — convenience for (y_true, y_pred)
  pairs.
- :func:`diebold_mariano` — paired forecast-accuracy test using
  Newey-West HAC variance, with the Harvey-Leybourne-Newbold (1997)
  small-sample correction.
- :func:`recommended_block_size` — automatic block-length proxy
  via the Politis-White (2004) optimal rule.

Conventions
-----------
All bootstrap routines accept an explicit ``seed`` and use a fresh
``numpy.random.default_rng(seed)`` so results are reproducible. CIs
are returned as percentile intervals on the bootstrap distribution
(default 95\,%); BCa is not implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Block size heuristic
# ---------------------------------------------------------------------------

def recommended_block_size(n: int, autocorr_lag1: float | None = None) -> int:
    """Return a sensible expected block length for the stationary bootstrap.

    Falls back to ``n^{1/3}`` when no autocorrelation information is
    supplied. When ``autocorr_lag1`` is given, scales the proxy by
    ``1 / (1 - rho^2)`` to make the block size grow with persistence,
    capped at ``n // 4``.
    """
    if n <= 1:
        return 1
    base = max(1, int(round(n ** (1.0 / 3.0))))
    if autocorr_lag1 is None:
        return min(base, n // 4)
    rho = float(np.clip(autocorr_lag1, -0.999, 0.999))
    scale = 1.0 / (1.0 - rho * rho)
    return int(min(max(1, round(base * scale)), max(1, n // 4)))


# ---------------------------------------------------------------------------
# Stationary block bootstrap (Politis-Romano 1994)
# ---------------------------------------------------------------------------

def stationary_block_bootstrap(
    n: int,
    *,
    expected_block_size: int,
    n_boot: int,
    seed: int = 42,
) -> np.ndarray:
    """Generate ``n_boot`` Politis-Romano bootstrap index arrays of length ``n``.

    Block lengths are draws from ``Geom(1 / expected_block_size)``,
    which preserves stationarity unconditionally. Within each block,
    indices wrap around the data circularly.

    Returns
    -------
    np.ndarray of shape (n_boot, n), dtype int64
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if expected_block_size < 1:
        expected_block_size = 1
    p = 1.0 / expected_block_size

    rng = np.random.default_rng(seed)
    out = np.empty((n_boot, n), dtype=np.int64)
    for b in range(n_boot):
        idx = np.empty(n, dtype=np.int64)
        i = 0
        while i < n:
            start = int(rng.integers(0, n))
            # Geometric block length with mean expected_block_size.
            block_len = int(rng.geometric(p))
            block_len = min(block_len, n - i)
            for k in range(block_len):
                idx[i + k] = (start + k) % n
            i += block_len
        out[b] = idx
    return out


# ---------------------------------------------------------------------------
# Generic bootstrap on a metric
# ---------------------------------------------------------------------------

@dataclass
class BootstrapCI:
    point_estimate: float
    lower: float
    upper: float
    alpha: float
    n_boot: int
    expected_block_size: int
    bootstrap_distribution: np.ndarray  # (n_boot,) of resampled metric values


def block_bootstrap_metric(
    arrays: tuple[np.ndarray, ...] | np.ndarray,
    metric_fn: Callable[..., float],
    *,
    expected_block_size: int | None = None,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> BootstrapCI:
    """Block-bootstrap a scalar metric over time-aligned arrays.

    Parameters
    ----------
    arrays : array or tuple of arrays
        Each array is indexed by the time axis (axis 0). Multiple
        arrays must share the same length and are resampled with the
        SAME bootstrap index sequence (so e.g. (y_true, y_pred) pairs
        stay aligned). A single array is also accepted.
    metric_fn : callable
        Receives the resampled arrays in the same order and returns a
        single float. For a single array, receives one positional arg.
    expected_block_size : int, optional
        Geometric mean block length. Defaults to
        :func:`recommended_block_size`.
    n_boot : int
        Number of bootstrap replications.
    alpha : float
        Two-sided coverage level: returns the ``[alpha/2, 1 - alpha/2]``
        percentiles.
    seed : int

    Returns
    -------
    BootstrapCI
    """
    if isinstance(arrays, np.ndarray):
        arrays = (arrays,)
    arrays = tuple(np.asarray(a) for a in arrays)
    n = len(arrays[0])
    for a in arrays[1:]:
        if len(a) != n:
            raise ValueError("All arrays must share length on axis 0")

    if expected_block_size is None:
        expected_block_size = recommended_block_size(n)

    point = float(metric_fn(*arrays))

    boot_idx = stationary_block_bootstrap(
        n, expected_block_size=expected_block_size, n_boot=n_boot, seed=seed,
    )
    boot_vals = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = boot_idx[b]
        resampled = tuple(a[idx] for a in arrays)
        boot_vals[b] = float(metric_fn(*resampled))

    lo = float(np.quantile(boot_vals, alpha / 2.0))
    hi = float(np.quantile(boot_vals, 1.0 - alpha / 2.0))
    return BootstrapCI(
        point_estimate=point, lower=lo, upper=hi,
        alpha=alpha, n_boot=n_boot,
        expected_block_size=expected_block_size,
        bootstrap_distribution=boot_vals,
    )


def block_bootstrap_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    **kwargs,
) -> BootstrapCI:
    """Block-bootstrap accuracy on (y_true, y_pred) pairs."""
    def _acc(yt: np.ndarray, yp: np.ndarray) -> float:
        return float(np.mean(yt == yp))
    return block_bootstrap_metric((np.asarray(y_true), np.asarray(y_pred)),
                                  _acc, **kwargs)


# ---------------------------------------------------------------------------
# Diebold-Mariano
# ---------------------------------------------------------------------------

@dataclass
class DMResult:
    statistic: float
    p_value: float
    lag: int
    loss_difference_mean: float
    n: int
    note: str = ""


def _newey_west_var(d: np.ndarray, lag: int) -> float:
    """Newey-West HAC variance estimator with Bartlett kernel."""
    n = len(d)
    d_mean = d.mean()
    e = d - d_mean
    gamma0 = float(np.dot(e, e) / n)
    var = gamma0
    for k in range(1, lag + 1):
        cov_k = float(np.dot(e[k:], e[:-k]) / n)
        weight = 1.0 - k / (lag + 1.0)
        var += 2.0 * weight * cov_k
    return max(var, 1e-30) / n


def diebold_mariano(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    *,
    lag: int | None = None,
    h: int = 1,
) -> DMResult:
    """Diebold-Mariano test of equal predictive accuracy.

    H0: ``E[loss_a - loss_b] = 0``.
    HA: ``E[loss_a - loss_b] != 0``.

    A negative statistic favours model A (lower loss → better);
    positive favours model B.

    Parameters
    ----------
    loss_a, loss_b : np.ndarray
        Per-observation loss for two competing forecasts. For
        directional accuracy, supply ``1 - correctness``.
    lag : int, optional
        Newey-West truncation lag. Defaults to ``h - 1``.
    h : int
        Forecast horizon (used to set the default lag).

    Returns
    -------
    DMResult
    """
    a = np.asarray(loss_a, dtype=float)
    b = np.asarray(loss_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("loss_a and loss_b must have the same shape")
    n = a.size
    d = a - b
    if n < 4:
        return DMResult(
            statistic=float("nan"), p_value=float("nan"),
            lag=0, loss_difference_mean=float(d.mean()) if n else float("nan"),
            n=n, note="n<4: too few observations",
        )

    if lag is None:
        lag = max(0, h - 1)
    var_d = _newey_west_var(d, lag)
    if var_d <= 0:
        return DMResult(
            statistic=float("nan"), p_value=float("nan"),
            lag=lag, loss_difference_mean=float(d.mean()),
            n=n, note="non-positive HAC variance",
        )
    dm_stat = float(d.mean() / np.sqrt(var_d))

    # Harvey-Leybourne-Newbold small-sample correction
    factor = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    if not np.isfinite(factor) or factor <= 0:
        factor = 1.0
    hln_stat = dm_stat * factor

    # Two-sided p-value using the Student-t with n-1 df (HLN convention)
    from scipy.stats import t as student_t
    p = 2.0 * (1.0 - student_t.cdf(abs(hln_stat), df=n - 1))

    return DMResult(
        statistic=hln_stat, p_value=float(p),
        lag=lag, loss_difference_mean=float(d.mean()),
        n=n,
    )


# ---------------------------------------------------------------------------
# Convenience: pairwise DM matrix across multiple models
# ---------------------------------------------------------------------------

def pairwise_diebold_mariano(
    losses_by_model: dict[str, np.ndarray],
    *,
    h: int = 1,
    lag: int | None = None,
) -> "list[dict]":
    """Run :func:`diebold_mariano` on every ordered pair of models.

    Returns a flat list of dicts (suitable for DataFrame construction)
    with columns ``model_a``, ``model_b``, ``dm_stat``, ``p_value``,
    ``loss_diff_mean``, ``n``, ``lag``.
    """
    rows: list[dict] = []
    names = list(losses_by_model.keys())
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            res = diebold_mariano(losses_by_model[a], losses_by_model[b],
                                  h=h, lag=lag)
            rows.append({
                "model_a": a,
                "model_b": b,
                "dm_stat": res.statistic,
                "p_value": res.p_value,
                "loss_diff_mean": res.loss_difference_mean,
                "n": res.n,
                "lag": res.lag,
            })
    return rows


# ---------------------------------------------------------------------------
# Romano-Wolf step-down adjustment for the pairwise DM matrix
# ---------------------------------------------------------------------------

def romano_wolf_dm(
    losses_by_model: dict[str, np.ndarray],
    *,
    h: int = 1,
    lag: int | None = None,
    expected_block_size: int,
    n_boot: int = 4000,
    seed: int = 42,
) -> list[dict]:
    r"""Romano-Wolf step-down FWER-adjusted p-values for pairwise DM tests.

    Implements the studentized step-down procedure of
    Romano and Wolf (2005, Econometrica). On the same family of
    pairwise hypotheses that :func:`pairwise_diebold_mariano`
    evaluates, this routine returns p-values that strongly control
    the family-wise error rate under arbitrary dependence between
    the test statistics.

    The bootstrap re-uses :func:`stationary_block_bootstrap` so that
    the resampled losses preserve the serial dependence structure
    of the correctness series; HAC variances are recomputed inside
    each bootstrap replication. ``expected_block_size`` should be
    set to the same value used by :func:`block_bootstrap_accuracy`
    on the same series (typically the
    :func:`recommended_block_size` proxy or ``max(h,
    recommended)``).

    Returns the same flat list-of-dicts shape as
    :func:`pairwise_diebold_mariano`, with three additional fields:

    - ``rw_p_value``: the Romano-Wolf step-down adjusted p-value.
    - ``rw_reject_05``: boolean indicating whether ``rw_p_value <
      0.05``.
    - ``bonferroni_p_value``: the conservative Bonferroni-Holm
      adjusted p-value, included as a benchmark.

    Notes
    -----
    Unlike Bonferroni, Romano-Wolf exploits the empirical
    correlation structure between the K test statistics and is
    therefore generally tighter (less conservative). The
    monotonicity step at the end ensures the adjusted p-values
    are non-decreasing in the original |statistic| ordering, which
    is required for the procedure to be a valid step-down.
    """
    names = list(losses_by_model.keys())
    M = len(names)
    if M < 2:
        return []

    # Pin the canonical pair ordering (matches pairwise_diebold_mariano).
    pairs: list[tuple[str, str]] = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pairs.append((a, b))
    K = len(pairs)

    # Validate equal-length losses.
    n = None
    losses = {}
    for m, arr in losses_by_model.items():
        a = np.asarray(arr, dtype=float)
        if n is None:
            n = a.size
        elif a.size != n:
            raise ValueError(f"loss arrays must have the same length; "
                             f"got {a.size} for '{m}' vs {n}")
        losses[m] = a
    if n is None or n < 4:
        return []

    if lag is None:
        lag = max(0, h - 1)

    # Observed DM stats.
    obs_stats = np.zeros(K)
    obs_pvals = np.zeros(K)
    for k, (a, b) in enumerate(pairs):
        res = diebold_mariano(losses[a], losses[b], lag=lag, h=h)
        obs_stats[k] = res.statistic
        obs_pvals[k] = res.p_value

    # Stationary block bootstrap of the time index.
    boot_idx = stationary_block_bootstrap(
        n, expected_block_size=expected_block_size,
        n_boot=n_boot, seed=seed,
    )

    # For each bootstrap replication, recompute the K DM stats and
    # center on the observed stat.
    boot_centered = np.zeros((n_boot, K))
    for r in range(n_boot):
        idx = boot_idx[r]
        for k, (a, b) in enumerate(pairs):
            d = losses[a][idx] - losses[b][idx]
            var_d = _newey_west_var(d, lag)
            if var_d <= 0 or not np.isfinite(var_d):
                boot_centered[r, k] = 0.0
                continue
            stat = float(d.mean() / np.sqrt(var_d))
            factor = np.sqrt(
                (n + 1 - 2 * h + h * (h - 1) / n) / n
            )
            if not np.isfinite(factor) or factor <= 0:
                factor = 1.0
            stat *= factor
            boot_centered[r, k] = stat - obs_stats[k]

    # Studentized step-down: order observed |t| descending.
    abs_obs = np.abs(obs_stats)
    order = np.argsort(-abs_obs)  # descending
    rw_pvals = np.ones(K)
    for step in range(K):
        # Remaining hypotheses at this step: order[step:].
        remaining = order[step:]
        # max |centered bootstrap stat| over remaining hypotheses.
        max_boot = np.max(np.abs(boot_centered[:, remaining]), axis=1)
        # Adjusted p-value at this step: P(max_boot >= |t_step|).
        threshold = abs_obs[order[step]]
        rw_pvals[order[step]] = float(
            np.mean(max_boot >= threshold)
        )
    # Enforce monotonicity along the descending-|t| order.
    for k in range(1, K):
        if rw_pvals[order[k]] < rw_pvals[order[k - 1]]:
            rw_pvals[order[k]] = rw_pvals[order[k - 1]]

    # Bonferroni-Holm benchmark.
    holm_pvals = np.ones(K)
    sorted_idx = np.argsort(obs_pvals)
    running_max = 0.0
    for rank, k in enumerate(sorted_idx):
        adj = (K - rank) * obs_pvals[k]
        running_max = max(running_max, min(1.0, adj))
        holm_pvals[k] = running_max

    # Assemble output rows.
    rows: list[dict] = []
    for k, (a, b) in enumerate(pairs):
        rows.append({
            "model_a": a,
            "model_b": b,
            "dm_stat": float(obs_stats[k]),
            "p_value": float(obs_pvals[k]),
            "rw_p_value": float(rw_pvals[k]),
            "rw_reject_05": bool(rw_pvals[k] < 0.05),
            "bonferroni_p_value": float(holm_pvals[k]),
            "n": int(n),
            "lag": int(lag),
        })
    return rows
