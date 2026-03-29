"""Lightweight: OLS R² of forward return on signal columns (research helper)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def ols_r2(X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    z = pd.concat([X, y.rename("y")], axis=1).dropna()
    if len(z) < 20:
        return {}
    Xm = z[X.columns].values
    ym = z["y"].values
    lr = LinearRegression().fit(Xm, ym)
    pred = lr.predict(Xm)
    ss_res = float(np.sum((ym - pred) ** 2))
    ss_tot = float(np.sum((ym - ym.mean()) ** 2)) + 1e-12
    return {"r2": 1.0 - ss_res / ss_tot}
