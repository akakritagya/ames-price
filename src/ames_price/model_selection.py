"""From-scratch k-fold index splitting for cross-validation."""

from __future__ import annotations

import numpy as np


def kfold_split(
    n_samples: int, n_splits: int = 5, random_state: int | None = None
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Shuffle 0..n_samples-1, then split into n_splits folds.

    Index-only: never touches X/y directly. Apply the returned indices
    to your own arrays/DataFrames (e.g. ``X_raw.iloc[train_idx]``).

    Parameters
    ----------
    n_samples : int
        Number of rows to split, over the index range range(n_samples).
    n_splits : int, default=5
        Number of folds.
    random_state : int or None, default=None
        Seed for the shuffle. None gives a different, non-reproducible
        shuffle on every call.

    Returns
    -------
    list of (train_idx, val_idx)
        One pair per fold, val_idx is that fold's held-out indices,
        train_idx is every other index. Folds partition
        range(n_samples) exactly once; sizes differ by at most one row
        when n_samples is not evenly divisible by n_splits (the first
        n_samples % n_splits folds get one extra row).
    """
    rng = np.random.default_rng(random_state)
    shuffled = rng.permutation(n_samples)

    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[: n_samples % n_splits] += 1

    folds = []
    start = 0
    for size in fold_sizes:
        val_idx = shuffled[start : start + size]
        train_idx = np.concatenate([shuffled[:start], shuffled[start + size :]])
        folds.append((train_idx, val_idx))
        start += size

    return folds
