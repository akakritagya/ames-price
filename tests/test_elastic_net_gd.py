import numpy as np

from ames_price.models.elastic_net_gd import ElasticNetGD


def _sparse_linear_fixture(
    n: int = 200, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    # x1, x2 are informative; x3, x4 have a true coefficient of exactly
    # zero -- same fixture as test_lasso_gd.py, reused here since it
    # exercises both recovery and sparsity in one dataset.
    true_coef = np.array([3.0, -2.0, 0.0, 0.0])
    true_intercept = 5.0
    noise = rng.normal(scale=0.01, size=n)
    y = X @ true_coef + true_intercept + noise
    return X, y, true_coef, true_intercept


def test_l1_ratio_one_matches_lasso_behavior():
    X, y, true_coef, true_intercept = _sparse_linear_fixture()
    model = ElasticNetGD(
        learning_rate=0.1,
        batch_size=20,
        n_epochs=300,
        alpha=0.5,
        l1_ratio=1.0,
        random_state=42,
    )
    model.fit(X, y)

    # at l1_ratio=1.0 the penalty gradient is bitwise identical to
    # LassoGD's -- same qualitative behavior as test_lasso_gd.py's
    # sparsity test at the same alpha.
    assert abs(model.coef_[2]) < 0.1
    assert abs(model.coef_[3]) < 0.1
    assert np.allclose(model.coef_[:2], true_coef[:2], atol=0.5)
    assert abs(model.intercept_ - true_intercept) < 0.1


def test_l1_ratio_half_shows_mixed_shrinkage():
    X, y, true_coef, true_intercept = _sparse_linear_fixture()
    model = ElasticNetGD(
        learning_rate=0.1,
        batch_size=20,
        n_epochs=300,
        alpha=0.5,
        l1_ratio=0.5,
        random_state=42,
    )
    model.fit(X, y)

    # the added L2 term biases every coefficient toward zero, not just
    # the true-zero ones -- looser tolerance than the pure-L1 case.
    assert np.allclose(model.coef_[:2], true_coef[:2], atol=0.8)
    # x3/x4 still shrink toward zero, though less tightly than l1_ratio=1.
    assert abs(model.coef_[2]) < 0.3
    assert abs(model.coef_[3]) < 0.3
    assert abs(model.intercept_ - true_intercept) < 0.1


def test_predict_output_shape_matches_input_rows():
    X, y, _, _ = _sparse_linear_fixture()
    model = ElasticNetGD(n_epochs=10, random_state=42)
    model.fit(X, y)

    X_new = np.random.default_rng(0).normal(size=(5, 4))
    predictions = model.predict(X_new)

    assert predictions.shape == (5,)


def test_loss_history_length_and_non_increasing():
    X, y, _, _ = _sparse_linear_fixture()
    model = ElasticNetGD(
        learning_rate=0.005,
        batch_size=len(X),
        n_epochs=200,
        alpha=0.5,
        l1_ratio=0.5,
        random_state=42,
    )
    model.fit(X, y)

    assert len(model.loss_history_) == 200
    assert all(
        model.loss_history_[i + 1] <= model.loss_history_[i] + 1e-9
        for i in range(len(model.loss_history_) - 1)
    )
