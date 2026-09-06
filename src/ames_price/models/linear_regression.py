"""Plain-numpy linear regression fit by mini-batch gradient descent."""

from __future__ import annotations

import numpy as np


class LinearRegressionGD:
    """Ordinary linear regression fit by mini-batch gradient descent.

    Weights start at zero, the intercept is tracked as its own scalar
    (not folded into X via a ones-column) so each parameter's gradient
    update stays an explicit, separately-explainable line. Fixed epoch
    count, no tolerance-based early stopping -- loss_history_ is meant to
    be plotted and n_epochs picked by eye from where it flattens.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        n_epochs: int = 100,
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
        random_state : int or None, default=None
            Seed for the per-epoch row shuffle. None gives a different,
            non-reproducible shuffle on every fit() call.
        verbose : bool, default=False
            If True, print the epoch number and loss every 10 epochs
            (plus the final epoch).
        """
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X: np.ndarray, y: np.ndarray) -> LinearRegressionGD:
        """Fit by mini-batch gradient descent.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        LinearRegressionGD
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
                grad_coef = (2 / len(batch_idx)) * (X_batch.T @ error)
                grad_intercept = (2 / len(batch_idx)) * error.sum()

                self.coef_ -= self.learning_rate * grad_coef
                self.intercept_ -= self.learning_rate * grad_intercept

            epoch_pred = X @ self.coef_ + self.intercept_
            epoch_loss = float(np.mean((epoch_pred - y) ** 2))
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
