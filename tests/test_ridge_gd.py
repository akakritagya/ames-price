import numpy as np

from ames_price.models.ridge_gd import RidgeGD


def _linear_fixture(
    n: int = 200, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    true_coef = np.array([3.0, -2.0])
    true_intercept = 5.0
    noise = rng.normal(scale=0.01, size=n)
    y = X @ true_coef + true_intercept + noise
    return X, y, true_coef, true_intercept


def test_fit_recovers_approximately_correct_coefficients():
    X, y, true_coef, true_intercept = _linear_fixture()
    model = RidgeGD(
        learning_rate=0.1,
        batch_size=20,
        n_epochs=300,
        alpha=0.1,
        random_state=42,
    )
    model.fit(X, y)

    # Ridge is intentionally biased toward smaller |coef_| than the true
    # weights -- a looser tolerance than plain OLS's, not a bug.
    assert np.allclose(model.coef_, true_coef, atol=0.3)
    # the intercept is never penalized, so it still recovers tightly.
    assert abs(model.intercept_ - true_intercept) < 0.1


def test_predict_output_shape_matches_input_rows():
    X, y, _, _ = _linear_fixture()
    model = RidgeGD(n_epochs=10, random_state=42)
    model.fit(X, y)

    X_new = np.random.default_rng(0).normal(size=(5, 2))
    predictions = model.predict(X_new)

    assert predictions.shape == (5,)


def test_loss_history_length_and_non_increasing():
    X, y, _, _ = _linear_fixture()
    model = RidgeGD(
        learning_rate=0.1,
        batch_size=len(X),
        n_epochs=200,
        alpha=0.1,
        random_state=42,
    )
    model.fit(X, y)

    assert len(model.loss_history_) == 200
    assert all(
        model.loss_history_[i + 1] <= model.loss_history_[i] + 1e-9
        for i in range(len(model.loss_history_) - 1)
    )
