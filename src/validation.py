"""Cross-validation utilities: Standard K-Fold, Purged K-Fold, temporal split."""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection._split import _BaseKFold


class PurgedKFold(_BaseKFold):
    """K-Fold CV with purging and embargo for financial time series.

    Follows Lopez de Prado, *Advances in Financial Machine Learning*, Ch. 7.

    Parameters
    ----------
    n_splits : int
    t1 : pd.Series — maps observation index to label end time.
    pct_embargo : float — fraction of observations to embargo after test sets.
    """

    def __init__(self, n_splits=5, t1=None, pct_embargo=0.01):
        if not isinstance(t1, pd.Series):
            raise ValueError("t1 must be a pandas Series")
        super().__init__(n_splits=n_splits, shuffle=False, random_state=None)
        self.t1 = t1
        self.pct_embargo = pct_embargo

    def split(self, X, y=None, groups=None):
        """Yield (train_indices, test_indices) with purging + embargo."""
        indices = np.arange(len(X))
        embargo_size = int(len(indices) * self.pct_embargo)
        test_starts = [(i * len(indices)) // self.n_splits for i in range(self.n_splits)]

        for i in range(self.n_splits):
            test_start = test_starts[i]
            test_end = test_starts[i + 1] if i < self.n_splits - 1 else len(indices)
            test_indices = indices[test_start:test_end]

            if hasattr(X, "index"):
                test_times = self.t1.loc[X.index[test_indices]]
                max_test_t1 = test_times.max()
                min_test_t0 = X.index[test_indices].min()
            else:
                test_times = self.t1.iloc[test_indices]
                max_test_t1 = test_times.max()
                min_test_t0 = test_indices.min()

            train_indices = []
            for j in indices:
                if j in test_indices:
                    continue

                if hasattr(X, "index"):
                    obs_t1 = self.t1.loc[X.index[j]]
                    obs_t0 = X.index[j]
                else:
                    obs_t1 = self.t1.iloc[j]
                    obs_t0 = j

                # Purge: label overlaps with test period
                if obs_t1 >= min_test_t0 and obs_t0 <= max_test_t1:
                    continue
                # Embargo: right after test set
                if j >= test_end and j < test_end + embargo_size:
                    continue

                train_indices.append(j)

            yield np.array(train_indices), test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


class WalkForwardCV:
    r"""Expanding-window walk-forward cross-validation for time series.

    Closes audit issue C-9. Unlike :class:`PurgedKFold`, walk-forward
    enforces strict temporal precedence: every observation in the
    training set occurs strictly before every observation in the test
    set, so no purging or embargo is needed. Successive outer folds
    expand the training window forward in time, yielding ``n_splits``
    chronologically-ordered (train, test) pairs.

    Parameters
    ----------
    n_splits : int
        Number of walk-forward steps.
    min_train_fraction : float
        Fraction of the series reserved as the initial training
        window (i.e. the smallest training set used). Must be in
        ``(0, 1)``.
    embargo : int
        Number of observations to drop between the training tail
        and the test head, to absorb any horizon-induced overlap
        ($h - 1$ at horizon $h$). Default 0 because, with strict
        chronological splits, the only overlap risk comes from the
        label horizon and is handled by the caller.
    """

    def __init__(self, n_splits: int = 5,
                 *, min_train_fraction: float = 0.4,
                 embargo: int = 0):
        if n_splits < 1:
            raise ValueError("n_splits must be >= 1")
        if not (0.0 < min_train_fraction < 1.0):
            raise ValueError("min_train_fraction must be in (0, 1)")
        self.n_splits = n_splits
        self.min_train_fraction = min_train_fraction
        self.embargo = max(0, int(embargo))

    def split(self, X, y=None, groups=None):
        n = len(X)
        if n < self.n_splits + 2:
            raise ValueError(
                f"too few observations ({n}) for n_splits={self.n_splits}"
            )
        first_train_end = int(round(n * self.min_train_fraction))
        first_train_end = max(1, min(first_train_end, n - self.n_splits))
        # Test slices partition the post-min-train tail into n_splits
        # equal-length chunks.
        remaining = n - first_train_end
        slice_size = remaining // self.n_splits
        if slice_size < 1:
            raise ValueError(
                f"min_train_fraction={self.min_train_fraction} leaves no "
                f"room for {self.n_splits} test slices in a series of "
                f"length {n}"
            )
        for i in range(self.n_splits):
            train_end = first_train_end + i * slice_size
            test_start = train_end + self.embargo
            test_end = test_start + slice_size
            if i == self.n_splits - 1:
                test_end = n  # absorb remainder into the last test slice
            if test_start >= n or test_end <= test_start:
                continue
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


def get_standard_kfold_splits(X, y, n_splits=5, random_state=42):
    """Yield (train_idx, test_idx) from standard (shuffled) K-Fold."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train_idx, test_idx in kf.split(X):
        yield train_idx, test_idx


def temporal_train_test_split(X, y, t1=None, test_size=0.2):
    """Chronological train/test split (no look-ahead bias).

    Parameters
    ----------
    X : pd.DataFrame
    y : pd.Series
    t1 : pd.Series, optional
    test_size : float

    Returns
    -------
    dict with keys X_train, X_test, y_train, y_test, split_date,
    and optionally t1_train, t1_test.
    """
    split_idx = int(len(X) * (1 - test_size))

    result = {
        "X_train": X.iloc[:split_idx],
        "X_test": X.iloc[split_idx:],
        "y_train": y.iloc[:split_idx],
        "y_test": y.iloc[split_idx:],
        "split_date": X.index[split_idx],
    }

    if t1 is not None:
        result["t1_train"] = t1.iloc[:split_idx]
        result["t1_test"] = t1.iloc[split_idx:]

    return result
