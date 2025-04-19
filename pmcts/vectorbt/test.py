import numpy as np
import pandas as pd


def test(
    portvalues: pd.DataFrame,
    positions: pd.DataFrame,
    exposures: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    vectorized backtesting of pair portfolio
    """
    if exposures is None:
        exposures = np.ones_like(positions)

    returns = positions * portvalues.pct_change().shift(-1)
    returns = returns * exposures
    ret = returns.sum(axis=1)
    pv = 100 * (1 + ret.cumsum())
    return pv
