"""Unit tests for the leakage-aware inference module.

Each test pins one piece of the contract used by audit issue C-22:

1. Block bootstrap on i.i.d. data is comparable to a Wilson CI.
2. Block bootstrap on autocorrelated data is *wider* than the i.i.d.
   asymptotic CI (this is the whole point of using a block bootstrap).
3. Diebold-Mariano rejects when one model is strictly better.
4. Diebold-Mariano does not reject when the two models have identical
   per-observation loss.
5. Bootstrap CI shrinks as ``n_boot`` grows.
6. Determinism: same seed → same bootstrap distribution.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference import (  # noqa: E402
    balanced_accuracy,
    block_bootstrap_accuracy,
    block_bootstrap_metric,
    brier_score,
    brier_skill_score,
    diebold_mariano,
    pairwise_diebold_mariano,
    recommended_block_size,
    romano_wolf_dm,
    stationary_block_bootstrap,
)


# ---------------------------------------------------------------------------
# 1. i.i.d. CI ≈ Wilson CI
# ---------------------------------------------------------------------------

def _wilson_ci(p: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    z = 1.959963984540054  # invnormal(0.975)
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def test_block_bootstrap_iid_close_to_wilson():
    rng = np.random.default_rng(0)
    n = 1500
    p_true = 0.51
    correct = (rng.uniform(size=n) < p_true).astype(int)
    y_true = np.zeros(n, dtype=int)
    y_pred = np.where(correct == 1, y_true, 1 - y_true)

    ci_boot = block_bootstrap_accuracy(
        y_true, y_pred,
        expected_block_size=1,    # i.i.d. by construction → block=1
        n_boot=4000, seed=0,
    )
    wilson_lo, wilson_hi = _wilson_ci(ci_boot.point_estimate, n)

    # Bootstrap CI half-width should be within ~30% of Wilson half-width.
    boot_hw = (ci_boot.upper - ci_boot.lower) / 2.0
    wilson_hw = (wilson_hi - wilson_lo) / 2.0
    assert 0.7 * wilson_hw < boot_hw < 1.3 * wilson_hw, (
        f"boot half-width {boot_hw:.4f} vs Wilson {wilson_hw:.4f}"
    )


# ---------------------------------------------------------------------------
# 2. Block bootstrap is wider on autocorrelated data
# ---------------------------------------------------------------------------

def test_block_bootstrap_wider_under_autocorrelation():
    """Build a series of correctness flags with strong AR(1) structure;
    confirm that a non-trivial block size yields a wider CI than
    block_size=1 (i.e., than the i.i.d. asymptotic estimator)."""
    rng = np.random.default_rng(1)
    n = 1500
    rho = 0.85
    z = np.zeros(n)
    z[0] = rng.normal()
    for t in range(1, n):
        z[t] = rho * z[t - 1] + rng.normal()
    correct = (z > 0).astype(int)
    y_true = np.zeros(n, dtype=int)
    y_pred = np.where(correct == 1, y_true, 1 - y_true)

    ci_iid = block_bootstrap_accuracy(
        y_true, y_pred,
        expected_block_size=1, n_boot=2000, seed=2,
    )
    ci_block = block_bootstrap_accuracy(
        y_true, y_pred,
        expected_block_size=20, n_boot=2000, seed=2,
    )
    iid_hw   = (ci_iid.upper - ci_iid.lower) / 2.0
    block_hw = (ci_block.upper - ci_block.lower) / 2.0
    assert block_hw > iid_hw * 1.3, (
        f"expected block CI to be at least 30% wider; "
        f"got iid={iid_hw:.4f}, block={block_hw:.4f}"
    )


# ---------------------------------------------------------------------------
# 3. DM rejects when A is strictly better
# ---------------------------------------------------------------------------

def test_diebold_mariano_rejects_when_a_dominates():
    rng = np.random.default_rng(3)
    n = 800
    # loss_a is uniformly smaller than loss_b
    loss_b = rng.uniform(0.4, 0.6, size=n)
    loss_a = loss_b - 0.05  # always 5pp lower per observation
    res = diebold_mariano(loss_a, loss_b, h=1)
    assert res.statistic < 0, "negative DM stat means A has lower loss"
    assert res.p_value < 0.001


# ---------------------------------------------------------------------------
# 4. DM does not reject when losses are identical
# ---------------------------------------------------------------------------

def test_diebold_mariano_no_rejection_under_equality():
    rng = np.random.default_rng(4)
    n = 800
    loss = rng.uniform(0.4, 0.6, size=n)
    # Both models have *exactly* the same loss per observation.
    res = diebold_mariano(loss, loss, h=1)
    # NaN p-value or a non-significant one is acceptable here.
    assert math.isnan(res.p_value) or res.p_value > 0.05


def test_diebold_mariano_noisy_equal_models():
    """When two models produce equal expected loss but with independent
    noise, DM should fail to reject at conventional levels in
    expectation. We test that the p-value is above 0.05 on this seed."""
    rng = np.random.default_rng(5)
    n = 800
    loss_a = rng.uniform(0.4, 0.6, size=n)
    loss_b = rng.uniform(0.4, 0.6, size=n)
    res = diebold_mariano(loss_a, loss_b, h=1)
    assert res.p_value > 0.05


# ---------------------------------------------------------------------------
# 5. Bootstrap distribution stabilises with n_boot
# ---------------------------------------------------------------------------

def test_bootstrap_ci_stabilises():
    rng = np.random.default_rng(6)
    n = 500
    x = rng.normal(size=n)

    def _mean(arr):
        return float(arr.mean())

    ci_500  = block_bootstrap_metric(x, _mean,
                                     expected_block_size=5,
                                     n_boot=500, seed=11)
    ci_4000 = block_bootstrap_metric(x, _mean,
                                     expected_block_size=5,
                                     n_boot=4000, seed=11)
    # Point estimate is identical (same data); CI shouldn't move much
    # but the larger bootstrap should give a stable percentile.
    assert ci_500.point_estimate == ci_4000.point_estimate
    spread_500 = ci_500.upper - ci_500.lower
    spread_4000 = ci_4000.upper - ci_4000.lower
    # The two should agree to within ~15%
    assert abs(spread_500 - spread_4000) / spread_4000 < 0.15


# ---------------------------------------------------------------------------
# 6. Determinism
# ---------------------------------------------------------------------------

def test_seeding_is_deterministic():
    n = 200
    indices_a = stationary_block_bootstrap(
        n, expected_block_size=4, n_boot=10, seed=99,
    )
    indices_b = stationary_block_bootstrap(
        n, expected_block_size=4, n_boot=10, seed=99,
    )
    np.testing.assert_array_equal(indices_a, indices_b)


# ---------------------------------------------------------------------------
# 7. Pairwise DM convenience
# ---------------------------------------------------------------------------

def test_pairwise_dm_matrix():
    rng = np.random.default_rng(7)
    n = 600
    losses = {
        "A": rng.uniform(0.4, 0.5, size=n),  # consistently lowest loss
        "B": rng.uniform(0.5, 0.6, size=n),
        "C": rng.uniform(0.55, 0.65, size=n),  # consistently highest loss
    }
    rows = pairwise_diebold_mariano(losses, h=1)
    assert len(rows) == 3  # A vs B, A vs C, B vs C
    by_pair = {(r["model_a"], r["model_b"]): r for r in rows}
    # A < B and A < C and B < C in mean loss
    assert by_pair[("A", "B")]["dm_stat"] < 0
    assert by_pair[("A", "C")]["dm_stat"] < 0
    assert by_pair[("B", "C")]["dm_stat"] < 0


# ---------------------------------------------------------------------------
# 8. Recommended block size heuristic
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n,expected_floor", [
    (8, 2), (27, 3), (1000, 9), (10000, 21),
])
def test_recommended_block_size_grows_with_n(n, expected_floor):
    bs = recommended_block_size(n)
    assert bs >= expected_floor - 1
    assert bs <= n // 4 + 1


def test_recommended_block_size_grows_with_persistence():
    bs_low  = recommended_block_size(1000, autocorr_lag1=0.0)
    bs_high = recommended_block_size(1000, autocorr_lag1=0.9)
    assert bs_high > bs_low


# ---------------------------------------------------------------------------
# Romano-Wolf step-down adjustment
# ---------------------------------------------------------------------------

def test_romano_wolf_dominance_survives_correction():
    """Under clear dominance, Romano-Wolf adjusted p-value still rejects.

    A: zero-loss; B,C: heavy positive loss (A dominates). Even after
    FWER control over three pairs the dominance result must survive.
    """
    rng = np.random.default_rng(0)
    n = 600
    A = rng.uniform(0, 0.05, n)
    B = rng.uniform(0.4, 0.6, n)
    C = rng.uniform(0.5, 0.7, n)
    rows = romano_wolf_dm(
        {"A": A, "B": B, "C": C},
        h=1, expected_block_size=5, n_boot=500, seed=0,
    )
    assert len(rows) == 3
    # The (A, B) and (A, C) pair must clear FWER 0.05 by a wide margin.
    by_pair = {(r["model_a"], r["model_b"]): r for r in rows}
    assert by_pair[("A", "B")]["rw_p_value"] < 0.05
    assert by_pair[("A", "C")]["rw_p_value"] < 0.05


def test_romano_wolf_no_signal_does_not_reject():
    """All-equal losses produce no rejection under FWER control."""
    rng = np.random.default_rng(1)
    n = 400
    L = rng.uniform(0, 1, n)
    rows = romano_wolf_dm(
        {"A": L, "B": L.copy(), "C": L.copy()},
        h=1, expected_block_size=4, n_boot=300, seed=1,
    )
    assert all(r["rw_p_value"] >= 0.05 for r in rows)


def test_romano_wolf_is_at_least_as_strict_as_raw():
    """Adjusted p-value should be >= raw p-value for every comparison.

    This is the defining property of an FWER-controlling step-down.
    """
    rng = np.random.default_rng(2)
    n = 500
    A = rng.uniform(0, 0.4, n)
    B = rng.uniform(0.05, 0.45, n)
    C = rng.uniform(0.10, 0.50, n)
    rows = romano_wolf_dm(
        {"A": A, "B": B, "C": C},
        h=1, expected_block_size=5, n_boot=500, seed=2,
    )
    for r in rows:
        assert r["rw_p_value"] >= r["p_value"] - 1e-9


def test_romano_wolf_monotone_along_t_order():
    """RW p-values are non-decreasing as |t| decreases (step-down property)."""
    rng = np.random.default_rng(3)
    n = 400
    A = rng.uniform(0, 0.5, n)
    B = rng.uniform(0.2, 0.7, n)
    C = rng.uniform(0.3, 0.8, n)
    D = rng.uniform(0.4, 0.9, n)
    rows = romano_wolf_dm(
        {"A": A, "B": B, "C": C, "D": D},
        h=1, expected_block_size=5, n_boot=400, seed=3,
    )
    # Sort rows by descending |dm_stat| and verify rw_p is non-decreasing.
    rows_sorted = sorted(rows, key=lambda r: -abs(r["dm_stat"]))
    rw_seq = [r["rw_p_value"] for r in rows_sorted]
    assert all(rw_seq[i] <= rw_seq[i + 1] + 1e-9
               for i in range(len(rw_seq) - 1))


def test_romano_wolf_tighter_than_bonferroni_under_correlation():
    """When pairs share data, Romano-Wolf should usually be tighter
    than Holm (Bonferroni-style) on the smallest p-value, because it
    accounts for the empirical correlation between the pair statistics.
    The inequality is statistical (not deterministic), so we use a
    margin and a fixed seed and only require the smallest RW p-value
    to be no larger than the corresponding Holm p-value.
    """
    rng = np.random.default_rng(4)
    n = 500
    base = rng.uniform(0, 1, n)
    A = base + rng.normal(0, 0.05, n)
    B = base + 0.30 + rng.normal(0, 0.05, n)
    C = base + 0.32 + rng.normal(0, 0.05, n)
    rows = romano_wolf_dm(
        {"A": A, "B": B, "C": C},
        h=1, expected_block_size=5, n_boot=500, seed=4,
    )
    rw_min = min(r["rw_p_value"] for r in rows)
    bonf_min = min(r["bonferroni_p_value"] for r in rows)
    assert rw_min <= bonf_min + 1e-9


# ---------------------------------------------------------------------------
# Class-imbalance-aware metrics: balanced accuracy, Brier score
# ---------------------------------------------------------------------------

def test_balanced_accuracy_perfect_prediction():
    yt = np.array([1, 1, -1, -1, 1, -1])
    yp = yt.copy()
    assert balanced_accuracy(yt, yp) == pytest.approx(1.0)


def test_balanced_accuracy_majority_class_predictor():
    """A constant 'always UP' predictor on imbalanced data must
    score 0.5 regardless of the imbalance level."""
    # 80 percent UP, 20 percent DOWN. Constant +1 predictor.
    yt = np.array([1] * 80 + [-1] * 20)
    yp = np.ones_like(yt)
    raw_acc = float(np.mean(yt == yp))
    assert raw_acc == pytest.approx(0.80)
    assert balanced_accuracy(yt, yp) == pytest.approx(0.50)


def test_balanced_accuracy_random_predictor_imbalanced():
    """On imbalanced data, a 50/50 random predictor's balanced
    accuracy is around 0.5 in expectation, while raw accuracy
    is around the majority-class rate."""
    rng = np.random.default_rng(0)
    n = 4000
    yt = np.where(rng.uniform(size=n) < 0.7, 1, -1)
    yp = rng.choice([-1, 1], size=n)
    bacc = balanced_accuracy(yt, yp)
    assert abs(bacc - 0.50) < 0.03


def test_brier_score_perfect_calibration():
    """Brier of probabilities that match true labels exactly is 0."""
    yt = np.array([1, -1, 1, -1])
    p  = np.array([1.0, 0.0, 1.0, 0.0])
    assert brier_score(yt, p) == pytest.approx(0.0)


def test_brier_score_uniform_05_predictor():
    """Brier of constant 0.5 on balanced labels is 0.25."""
    yt = np.array([1, -1, 1, -1])
    p  = np.full(4, 0.5)
    assert brier_score(yt, p) == pytest.approx(0.25)


def test_brier_skill_score_zero_for_base_rate_predictor():
    """A constant predictor at the empirical base rate has BSS = 0."""
    yt = np.array([1] * 70 + [-1] * 30)
    p = np.full(len(yt), 0.7)
    assert abs(brier_skill_score(yt, p)) < 1e-9


def test_brier_skill_score_positive_when_better_than_base_rate():
    """A predictor better than base-rate has positive BSS."""
    yt = np.array([1] * 70 + [-1] * 30)
    # Closer to truth than the base rate
    p = np.where(yt == 1, 0.9, 0.1)
    assert brier_skill_score(yt, p) > 0.0
