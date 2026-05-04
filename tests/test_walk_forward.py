"""Unit tests for WalkForwardCV (audit C-9)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.validation import WalkForwardCV  # noqa: E402


@pytest.fixture
def linear_series():
    n = 200
    X = pd.DataFrame({"x": np.arange(n)}, index=pd.RangeIndex(n))
    y = pd.Series(np.arange(n), index=X.index)
    return X, y


def test_walk_forward_yields_n_splits(linear_series):
    X, y = linear_series
    wf = WalkForwardCV(n_splits=5, min_train_fraction=0.4)
    splits = list(wf.split(X, y))
    assert len(splits) == 5


def test_walk_forward_strict_temporal_precedence(linear_series):
    """Every train index must be strictly less than every test index."""
    X, y = linear_series
    wf = WalkForwardCV(n_splits=5, min_train_fraction=0.4)
    for train_idx, test_idx in wf.split(X, y):
        assert train_idx.max() < test_idx.min(), (
            f"train.max()={train_idx.max()} must be < "
            f"test.min()={test_idx.min()}"
        )


def test_walk_forward_train_is_expanding(linear_series):
    """Successive training-window sizes must be non-decreasing."""
    X, y = linear_series
    wf = WalkForwardCV(n_splits=5, min_train_fraction=0.4)
    sizes = [len(tr) for tr, _ in wf.split(X, y)]
    assert sizes == sorted(sizes)
    assert sizes[-1] > sizes[0]


def test_walk_forward_test_partitions_tail(linear_series):
    """The union of all test slices covers the post-min-train tail."""
    X, y = linear_series
    n = len(X)
    wf = WalkForwardCV(n_splits=5, min_train_fraction=0.4)
    splits = list(wf.split(X, y))
    test_indices = np.concatenate([t for _, t in splits])
    # Min-train end is at round(0.4 * 200) = 80.
    first_train_end = int(round(n * 0.4))
    assert test_indices.min() >= first_train_end
    assert test_indices.max() == n - 1


def test_walk_forward_embargo_drops_observations(linear_series):
    """An embargo of k drops k observations between train tail and test head."""
    X, y = linear_series
    wf = WalkForwardCV(n_splits=5, min_train_fraction=0.4, embargo=3)
    for train_idx, test_idx in wf.split(X, y):
        gap = test_idx.min() - train_idx.max() - 1
        assert gap >= 3


def test_walk_forward_too_few_splits_raises():
    X = pd.DataFrame({"x": [1.0, 2.0]}, index=pd.RangeIndex(2))
    wf = WalkForwardCV(n_splits=5, min_train_fraction=0.4)
    with pytest.raises(ValueError, match="too few observations"):
        list(wf.split(X))


def test_walk_forward_invalid_min_train_fraction():
    with pytest.raises(ValueError, match="min_train_fraction"):
        WalkForwardCV(n_splits=5, min_train_fraction=1.0)
    with pytest.raises(ValueError, match="min_train_fraction"):
        WalkForwardCV(n_splits=5, min_train_fraction=0.0)
