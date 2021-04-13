import numpy as np
import pandas as pd
from typing import Tuple
from scipy import stats as st
from tools.helpers import validate_conditions_for_theoretical_distns


def single_mean_interval(sample: pd.Series, ci: float) -> Tuple[float, float]:
    """

    Args:
        sample: Numeric variable with the values in a Pandas Series
        ci: Level of confidence for the interval as a real number between 0 and 1. i.e. 0.90 for a 90% interval

    Returns:
        Tuple with the start and end values of the interval.
    """
    _statistics = sample.describe()
    _SE = _statistics['std'] / np.sqrt(_statistics['count'])
    df = _statistics['count'] - 1
    t_star = st.t.ppf((1-ci)/2, df=df)
    _ME = t_star * _SE
    validate_conditions_for_theoretical_distns(inference_type='single-mean', n=_statistics['count'])
    return _statistics['mean'] - _ME, _statistics['mean'] + _ME


def single_proportion_interval():
    # TODO: Implement. Feel free to create a pull request if you want to contribute.
    raise NotImplementedError


def two_means_interval():
    # TODO: Implement. Feel free to create a pull request if you want to contribute.
    raise NotImplementedError


def two_proportions_interval(sample: pd.Series, categories: Tuple[str, str], ci: float):
    """
    Args:
        sample: Series with the count of two categorical variables. Check the example below for details.
        categories: A Tuple with the name of the categories required to use the test. The order is important, the first
                    element of the Tuple will be used for `p1` and the second element for `p2`. The confidence interval
                    is one `p1 - p2`
        ci: Level of confidence for the interval as a real number between 0 and 1. i.e. 0.90 for a 90% interval

    Returns:
        Tuple with the start and end values of the interval.

    Example:
        The following is an example of the format for the `sample` parameter. Similarly to `single_proportion_interval`,
        the index values (Desipramine, Lithium, Placebo) are the categories, and the values are the count of elements in
        each category. The `two_proportions_interval`, requires at least three categories::

            >>> sample
            Out[1]:
            Drug
            Desipramine    14
            Lithium         6
            Placebo         4
            Name: Relapse, dtype: int64

    """
    n = sample.sum()
    n1 = sample[categories[0]]
    n2 = sample[categories[1]]
    p1_hat = n1 / n
    p2_hat = n2 / n
    _SE = np.sqrt((p1_hat*(1-p1_hat))/n1 + (p2_hat*(1-p2_hat))/n2)
    z_star = st.norm.ppf((1-ci)/2)
    _ME = z_star * _SE
    validate_conditions_for_theoretical_distns(inference_type='two-proportions', x1=(n1, p1_hat), x2=(n2, p2_hat))
    return (p1_hat-p2_hat) - _ME, (p1_hat-p2_hat) + _ME
