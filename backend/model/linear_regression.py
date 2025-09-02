"""
Simple Linear Regression implemented from scratch using NumPy.
Compatible with scikit-learn Pipelines (fit/predict, get_params/set_params).
"""
from __future__ import annotations

import numpy as np
from typing import Any, Dict


class MyLinearRegression:
    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0

    def get_params(self, deep: bool = True) -> Dict[str, Any]:  # sklearn API compat
        return {"fit_intercept": self.fit_intercept}

    def set_params(self, **params) -> "MyLinearRegression":  # sklearn API compat
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def _to_numpy(self, X):
        # Accept numpy arrays, pandas DataFrames, and scipy sparse matrices
        if hasattr(X, "toarray"):
            X = X.toarray()
        elif hasattr(X, "values"):
            X = X.values
        return np.asarray(X, dtype=float)

    def fit(self, X, y):
        X_np = self._to_numpy(X)
        y_np = np.asarray(y, dtype=float).ravel()

        if self.fit_intercept:
            ones = np.ones((X_np.shape[0], 1), dtype=float)
            X_ext = np.hstack([X_np, ones])
        else:
            X_ext = X_np

        # Solve least squares using a numerically-stable method
        beta, *_ = np.linalg.lstsq(X_ext, y_np, rcond=None)

        if self.fit_intercept:
            self.coef_ = beta[:-1]
            self.intercept_ = float(beta[-1])
        else:
            self.coef_ = beta
            self.intercept_ = 0.0

        return self

    def predict(self, X):
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted yet.")
        X_np = self._to_numpy(X)
        return X_np.dot(self.coef_) + self.intercept_
