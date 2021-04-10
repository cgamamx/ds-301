import numpy as np
import pandas as pd
from typing import Tuple
from scipy import stats as st


def single_mean_interval(sample: pd.Series, ci: float) -> Tuple[float, float]:
    _statistics = sample.describe()
    _SE = _statistics['std'] / np.sqrt(_statistics['count'])
    df = _statistics['count'] - 1
    t_star = st.t.ppf((1-ci)/2, df=df)
    _ME = t_star * _SE
    return _statistics['mean'] - _ME, _statistics['mean'] + _ME


def single_proportion_interval():
    # TODO: Implement
    raise NotImplementedError


def two_means_interval():
    # TODO: Implement
    raise NotImplementedError


def two_proportions_interval(sample: pd.Series, categories: Tuple[str, str], ci: float):
    n = sample.sum()
    n1 = sample[categories[0]]
    n2 = sample[categories[1]]
    p1_hat = n1 / n
    p2_hat = n2 / n
    _SE = np.sqrt((p1_hat*(1-p1_hat))/n1 + (p2_hat*(1-p2_hat))/n2)
    z_star = st.norm.ppf((1-ci)/2)
    _ME = z_star * _SE
    return (p1_hat-p2_hat) - _ME, (p1_hat-p2_hat) + _ME
