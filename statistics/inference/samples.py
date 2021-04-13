"""This module includes two functions to calculate the sample size for estimating one proportion and one mean. The
functions return the calculation as per the formula sheet provided by Dr. Parker, and are easy enough that they don't
require detailed documentation.
"""

from scipy import stats as st
import numpy as np


def single_proportion_sample_size(p_tilde: float = 0.5, margin: float = 0.05, confidence_interval: float = 0.95) -> int:
    _ME = margin
    z_star = st.norm.ppf((1-confidence_interval)/2)
    n = (z_star/_ME)**2 * p_tilde * (1-p_tilde)
    return np.ceil(n)


def single_mean_sample_size(sigma_tilde: float, margin: float = 0.05, confidence_interval: float = 0.95) -> int:
    _ME = margin
    z_star = st.norm.ppf((1-confidence_interval)/2)
    n = (z_star*sigma_tilde / _ME) ** 2
    return np.ceil(n)
