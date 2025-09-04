from __future__ import annotations

from typing import Dict, List, Sequence
import numpy as np
import pandas as pd


class Preprocessor:
    def __init__(self, cat_cols: Sequence[str], num_cols: Sequence[str]):
        self.cat_cols: List[str] = list(cat_cols)
        self.num_cols: List[str] = list(num_cols)
        self.categories_: Dict[str, List[str]] = {}
        self.num_means_: np.ndarray | None = None
        self.num_stds_: np.ndarray | None = None

    def fit(self, X: pd.DataFrame) -> "Preprocessor":
        # Fit categorical categories in a stable order
        for c in self.cat_cols:
            vals = X[c].astype(str).fillna("")
            cats = sorted(pd.Series(vals.unique()).astype(str))
            self.categories_[c] = cats
        # Fit numeric mean/std
        num = X[self.num_cols].astype(float).to_numpy(copy=False)
        self.num_means_ = np.nanmean(num, axis=0)
        std = np.nanstd(num, axis=0)
        # Avoid divide by zero
        std[std == 0] = 1.0
        self.num_stds_ = std
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        assert self.num_means_ is not None and self.num_stds_ is not None, "Preprocessor not fitted"
        # One-hot encode categorical
        ohe_parts: List[np.ndarray] = []
        for c in self.cat_cols:
            cats = self.categories_.get(c, [])
            idx_map = {v: i for i, v in enumerate(cats)}
            col_vals = X[c].astype(str).fillna("")
            mat = np.zeros((len(X), len(cats)), dtype=float)
            for row_i, v in enumerate(col_vals):
                j = idx_map.get(str(v))
                if j is not None:
                    mat[row_i, j] = 1.0
            ohe_parts.append(mat)
        # Standardize numeric
        num = X[self.num_cols].astype(float).to_numpy(copy=False)
        Znum = (num - self.num_means_) / self.num_stds_
        return np.hstack(parts)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)


class ScratchPipeline:
    def __init__(self, preprocessor: Preprocessor, regressor):
        self.preprocessor = preprocessor
        self.regressor = regressor

    def fit(self, X: pd.DataFrame, y: Sequence[float]):
        Z = self.preprocessor.fit_transform(X)
        self.regressor.fit(Z, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Z = self.preprocessor.transform(X)
        return self.regressor.predict(Z)
