import numpy as np
import pandas as pd
from typing import Dict, Tuple
from scipy import stats as ss
from tools.helpers import validate_conditions_for_theoretical_distns


def get_p_value(ha_parameter: float, distribution: str = 'norm', alternative: str = 'less', **kwargs) -> float:
    """Calculates the p-value for a test

    Args:
        ha_parameter: Value of the Alternate Hypothesis parameter
        distribution: Theoretical distribution to use, currently it only supports 't' or 'norm'
        alternative: Defines the alternative hypothesis. Possible values: 'less' (default), 'greater', or 'two-sided'.
        **kwargs: All the extra parameters will be packed in kwargs, i.e. df for degrees of freedom in the "t" dist

    Returns:
        The calculated p-value
    """
    if distribution == 't':
        dist = ss.t
    else:
        # Normal dist by default
        dist = ss.norm
    if alternative == 'greater':
        p_value = 1 - dist.cdf(ha_parameter, **kwargs)
    elif alternative == 'less':
        p_value = dist.cdf(ha_parameter, **kwargs)
    elif alternative == 'two-sided':
        # Two-sided only makes sense when two variables are involved. i.e. Difference of means or proportions, slope,
        # correlation, etc. Otherwise, the computed result will be greater than 1, and given that the p-value is a
        # probability, it should be less or equal than 1.
        p_value = np.minimum(2 * dist.cdf(-1 * np.abs(ha_parameter), **kwargs), 1)
    else:
        raise ValueError("alternative must be 'less', 'greater' or 'two-sided'")
    return p_value


def single_mean_test(sample: pd.Series, mu_0: float, alternative: str) -> Dict[str, float]:
    """Performs a single mean test

    Args:
        sample: Numeric variable with the values in a Pandas Series
        mu_0: Mean from the Null Hypothesis
        alternative: Defines the alternative hypothesis. Possible values: 'less', 'greater', or 'two-sided'.

    Returns:
        Dict with the calculated "t" parameter and the p-value
    """
    _statistics = sample.describe()
    _SE = _statistics['std'] / np.sqrt(_statistics['count'])
    t = (_statistics['mean'] - mu_0) / _SE
    df = _statistics['count'] - 1
    validate_conditions_for_theoretical_distns(inference_type='single-mean', n=_statistics['count'])
    return {'t': t, 'p-value': get_p_value(t, distribution='t', alternative=alternative, df=df)}


def single_proportion_test(sample: pd.Series, category: str, p_0: float, alternative: str) -> Dict[str, float]:
    """Performs a single proportion test

    Args:
        sample: Series with the count of two categorical variables. Check the example below for details.
        category: The name of the category we want to use for the test.
        p_0: The proportion of the Null Hypothesis
        alternative: Defines the alternative hypothesis. Possible values: 'less', 'greater', or 'two-sided'.

    Returns:
        Dict with the calculated "z" parameter and the p-value

    Example:
        The following is an example of the format required for the `sample` parameter. The index values (yes, no) are
        the categories, and the values are the count of elements in each category::

            >>> sample
            Out[1]:
            Relapse
            no      4
            yes    20
            Name: Drug, dtype: int64
    """
    n = sample.sum()
    p_hat = sample[category] / n
    _SE = np.sqrt(p_0 * (1 - p_0) / n)
    z = p_hat - p_0 / _SE
    validate_conditions_for_theoretical_distns(inference_type='single-proportion', n=n, p=p_hat)
    return {'z': z, 'p-value': get_p_value(z, alternative=alternative)}


def two_mean_test(data_stats: pd.DataFrame, categories: Tuple[str, str], alternative: str, **args) -> Dict[str, float]:
    """Performs a two mean test

    Args:
        data_stats: Summaary Statistics of two numerical variables. Check the example below for details.
        categories: A Tuple with the name of the categories required to use the test. The order is important, the first
            element of the Tuple will be `X1` and the second element will be `X2`. The Null Hypothesis is `X1 - X2 = 0`
        alternative: 'less', 'greater', or 'two-sided'.
        **args: Optional parameters will be packed in `args`. Currently, there is only one optional parameter:
            `df`, which is used to specify how to calculate the degrees of freedom for the t-distribution. If the value
            of df is set to `satterthwait`, the Satterthwait approximation will be used. Any other values will result in
            the minimum of `n1-1` or `n2-1`

    Returns:
        Dict with the calculated "t" parameter and the p-value

    Example:
        The following is an example of the format required for the `data_stats` parameter. The index values are the
        descriptive statistics of the numeric variable and it must include `count`, `mean`, and `std`. It can easily be
        generated with the `pandas.DataFrame.describe` method::

            >>> data_stats
            Out[1]:
            Drink     Coffee        Tea
            count  10.000000  11.000000
            mean   17.700000  34.818182
            std    16.693645  21.084678
            ...
            max    52.000000  58.000000

    """
    set_a = data_stats.get(categories[0])
    set_b = data_stats.get(categories[1])
    s1 = set_a['std']
    s2 = set_b['std']
    n1 = set_a['count']
    n2 = set_b['count']
    _SE = np.sqrt((s1 ** 2) / n1 + (s2 ** 2) / n2)
    t = (set_a['mean'] - set_b['mean']) / _SE

    if args.get('df') == 'satterthwait':
        df = ((s1 ** 2 / n1 + s1 ** 2 / n1) ** 2) / (
                    (1 / (n1 - 1)) * (s1 ** 2 / n1) ** 2 + (1 / (n2 - 1)) * (s2 ** 2 / n2) ** 2)
    else:
        df = min(n1 - 1, n2 - 1)

    validate_conditions_for_theoretical_distns(inference_type='two-means', n1=n1, n2=n2)
    return {'t': t, 'p-value': get_p_value(t, distribution='t', df=df, alternative=alternative)}


def two_proportions_test(sample: pd.Series, categories: Tuple[str, str], alternative: str) -> Dict[str, float]:
    """Performs a two proportions test

    Args:
        sample: Series with the count of two categorical variables. Check the example below for details.
        categories: A Tuple with the name of the categories required to use the test. The order is important, the first
            element of the Tuple will be used for `p1` and the second element for `p2`. The Null Hypothesis is
            `X1 - X2 = 0`
        alternative: 'less', 'greater', or 'two-sided'.

    Returns:
        Dict with the calculated "z" parameter and the p-value

    Example:
        The following is an example of the format for the `sample` parameter. Similarly to `single_proportion_test`, the
        index values (Desipramine, Lithium, Placebo) are the categories, and the values are the count of elements in
        each category. The `two_proportions_test`, requires at least three categories::

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
    p_bar = (n1 + n2) / (2 * n)
    _SE = np.sqrt((p_bar * (1 - p_bar)) / n + (p_bar * (1 - p_bar)) / n)
    z = (p1_hat - p2_hat) / _SE
    validate_conditions_for_theoretical_distns(inference_type='two-proportions', x1=(n1, p1_hat), x2=(n2, p2_hat))
    return {'z': z, 'p-value': get_p_value(z, alternative=alternative)}
