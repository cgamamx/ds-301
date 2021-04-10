import numpy as np
import pandas as pd
from typing import Dict, Tuple
from scipy import stats as st


def get_p_value(distance: float, distribution: str = 'norm', tail: str = 'left', **kwargs) -> float:
    if distribution == 't':
        dist = st.t
    else:
        # Normal dist by default
        dist = st.norm
    if tail == 'right':
        p_value = dist.cdf(-distance, **kwargs)
    elif tail == 'both':
        p_value = dist.cdf(distance, **kwargs) + (1 - dist.cdf(-distance, **kwargs))
    else:
        # Left tail by default
        p_value = dist.cdf(distance, **kwargs)
    return p_value


def single_mean_test(sample: pd.Series, mu_0: float, tail: str) -> Dict[str, float]:
    _statistics = sample.describe()
    _SE = _statistics['std'] / np.sqrt(_statistics['count'])
    t = (_statistics['mean'] - mu_0) / _SE
    df = _statistics['count'] - 1
    return {'t': t, 'p_value': get_p_value(t, distribution='t', tail=tail, df=df)}


def single_proportion_test(sample: pd.Series, category: str, p_0: float, tail: str) -> Dict[str, float]:
    n = sample.sum()
    p_hat = sample[category] / n
    _SE = np.sqrt(p_0 * (1 - p_0) / n)
    z = p_hat - p_0 / _SE
    return {'z': z, 'p_value': get_p_value(z, tail=tail)}


def two_mean_test(data_stats: pd.DataFrame, categories: Tuple[str, str], **args) -> Dict[str, float]:
    set_a = data_stats.get(categories[0])
    set_b = data_stats.get(categories[1])
    s1 = set_a['std']
    s2 = set_b['std']
    n1 = set_a['count']
    n2 = set_b['count']
    _SE = np.sqrt((s1**2)/n1 + (s2**2)/n2)
    t = (set_a['mean'] - set_b['mean']) / _SE

    if args.get('df') == 'satterthwait':
        df = ((s1**2/n1 + s1**2/n1)**2) / ((1/(n1-1))*(s1**2/n1)**2 + (1/(n2-1))*(s2**2/n2)**2)
    else:
        df = min(n1, n2)
    tail = args.get('tail', 'left')
    return {'t': t, 'p_value': get_p_value(t, distribution='t', df=df, tail=tail)}


def two_proportions_test(sample: pd.Series, categories: Tuple[str, str], tail: str) -> Dict[str, float]:
    n = sample.sum()
    n1 = sample[categories[0]]
    n2 = sample[categories[1]]
    p1_hat = n1 / n
    p2_hat = n2 / n
    p_bar = (n1+n2) / (2*n)
    _SE = np.sqrt((p_bar*(1-p_bar))/n + (p_bar*(1-p_bar))/n)
    z = (p1_hat - p2_hat) / _SE
    return {'z': z, 'p_value': get_p_value(z, tail=tail)}
