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
