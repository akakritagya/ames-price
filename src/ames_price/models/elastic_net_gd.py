"""ElasticNet-regularized linear regression via mini-batch GD."""

from __future__ import annotations

import numpy as np


class ElasticNetGD:
    """Linear regression with combined L1+L2 penalty using mini-batch GD.

    Same skeleton as ``LinearRegressionGD``/``RidgeGD``/``LassoGD``:
    weights start at zero, the intercept is tracked as its own scalar,
    and the batch loop is identical. The penalty is
    ``alpha * (l1_ratio * L1 + 0.5 * (1 - l1_ratio) * L2)`` where
    ``L1 = sum(abs(coef_))`` and ``L2 = sum(coef_**2)``, matching
    scikit-learn's ``ElasticNet`` parameterization -- ``alpha``
    is the overall penalty strength, ``l1_ratio`` in ``[0, 1]`` is the
    mixing weight between the L1 and L2 terms. Excluded from the
    intercept's update, same reasoning as ``RidgeGD``/``LassoGD``.

    At ``l1_ratio=1.0`` the penalty gradient (``alpha * np.sign(coef_)``)
    is bitwise identical to ``LassoGD``'s at the same ``alpha``. At
    ``l1_ratio=0.0`` it is ``alpha * coef_``, which is **not**
    numerically equal to ``RidgeGD``'s ``2 * alpha * coef_`` at the same
    ``alpha`` (a 2x difference in effective L2 strength) -- this is
    expected, not a bug, consistent with this project's general stance
    (see ``2_regularized_regression.ipynb``) that alpha is not
    comparable 1:1 across model families.

    Like ``LassoGD``, the L1 component uses the subgradient approximation
    (``np.sign(0) == 0``) rather than a proximal/soft-thresholding
    solver, so sparsity is soft: coefficients shrink toward zero and
    settle very close to it, but "near zero" (checked with a tolerance),
    not bitwise ``0.0``, is the correct thing to assert or expect.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        n_epochs: int = 100,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        random_state: int | None = None,
        verbose: bool = False,
    ) -> None:
        """Store hyperparameters; fitting happens in fit(), not here.

        Parameters
        ----------
        learning_rate : float, default=0.01
            Step size applied to each gradient update.
        batch_size : int, default=32
            Number of rows per mini-batch. A value >= n_samples degrades
            to full-batch gradient descent.
        n_epochs : int, default=100
            Number of passes over the shuffled training data.
        alpha : float, default=1.0
            Overall penalty strength applied to coef_. The intercept is
            never penalized.
        l1_ratio : float, default=0.5
            Mixing weight between the L1 and L2 penalty terms, in
            [0, 1]. 1.0 is pure L1 (matches LassoGD's gradient exactly),
            0.0 is pure L2 (shaped like RidgeGD's but not numerically
            equal to it -- see class docstring).
        random_state : int or None, default=None
            Seed for the per-epoch row shuffle. None gives a different,
            non-reproducible shuffle on every fit() call.
        verbose : bool, default=False
            If True, print the epoch number and penalized loss every 10
            epochs (plus the final epoch).
        """
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X: np.ndarray, y: np.ndarray) -> ElasticNetGD:
        """Fit using mini-batch GD on the ElasticNet-penalized loss.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        ElasticNetGD
            The fitted estimator, with coef_, intercept_, and
            loss_history_ set.
        """
        n_samples, n_features = X.shape
        rng = np.random.default_rng(self.random_state)

        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0.0
        self.loss_history_: list[float] = []

        for epoch in range(self.n_epochs):
            shuffled_idx = rng.permutation(n_samples)
            for start in range(0, n_samples, self.batch_size):
                batch_idx = shuffled_idx[start : start + self.batch_size]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]

                error = X_batch @ self.coef_ + self.intercept_ - y_batch
                mse_grad = (2 / len(batch_idx)) * (X_batch.T @ error)
                l1_l2_grad = self.alpha * (
                    self.l1_ratio * np.sign(self.coef_)
                    + (1 - self.l1_ratio) * self.coef_
                )
                grad_coef = mse_grad + l1_l2_grad
                grad_intercept = (2 / len(batch_idx)) * error.sum()

                self.coef_ -= self.learning_rate * grad_coef
                self.intercept_ -= self.learning_rate * grad_intercept

            epoch_pred = X @ self.coef_ + self.intercept_
            epoch_mse = float(np.mean((epoch_pred - y) ** 2))
            epoch_l1 = float(np.sum(np.abs(self.coef_)))
            epoch_l2 = float(np.sum(self.coef_**2))
            epoch_penalty = self.alpha * (
                self.l1_ratio * epoch_l1 + 0.5 * (1 - self.l1_ratio) * epoch_l2
            )
            epoch_loss = epoch_mse + epoch_penalty
            self.loss_history_.append(epoch_loss)

            if self.verbose and (epoch % 10 == 0 or epoch == self.n_epochs - 1):
                print(f"epoch {epoch + 1}/{self.n_epochs} - loss: {epoch_loss:.4f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values for X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Samples to predict on.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted values, X @ coef_ + intercept_.
        """
        return X @ self.coef_ + self.intercept_
