import numpy as np

from ames_price.model_selection import kfold_split


def test_folds_partition_all_indices_exactly_once():
    folds = kfold_split(n_samples=23, n_splits=5, random_state=42)

    all_val_idx = np.concatenate([val_idx for _, val_idx in folds])
    assert sorted(all_val_idx.tolist()) == list(range(23))


def test_train_and_val_disjoint_within_each_fold():
    folds = kfold_split(n_samples=23, n_splits=5, random_state=42)

    for train_idx, val_idx in folds:
        assert set(train_idx.tolist()).isdisjoint(set(val_idx.tolist()))
        assert len(train_idx) + len(val_idx) == 23


def test_fold_sizes_as_equal_as_possible():
    folds = kfold_split(n_samples=23, n_splits=5, random_state=42)

    # 23 = 5*4 + 3 -- three folds get 5 rows, two folds get 4.
    val_sizes = sorted(len(val_idx) for _, val_idx in folds)
    assert val_sizes == [4, 4, 5, 5, 5]


def test_fold_sizes_all_equal_when_evenly_divisible():
    folds = kfold_split(n_samples=20, n_splits=5, random_state=42)

    assert [len(val_idx) for _, val_idx in folds] == [4, 4, 4, 4, 4]


def test_same_random_state_reproduces_identical_folds():
    folds_a = kfold_split(n_samples=23, n_splits=5, random_state=7)
    folds_b = kfold_split(n_samples=23, n_splits=5, random_state=7)

    for (train_a, val_a), (train_b, val_b) in zip(folds_a, folds_b, strict=True):
        np.testing.assert_array_equal(train_a, train_b)
        np.testing.assert_array_equal(val_a, val_b)
