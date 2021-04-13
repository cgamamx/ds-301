from typing import Tuple


class TheoreticalConditionNotMetWarning:
    def __init__(self, message: str):
        self.message = message
        # This is the escape sequence to print to the terminal in red
        self._begin = '\033[91m'
        # End escape sequence
        self._end = '\033[0m'
        print(f'{self._begin}{self.message}{self._end}')


def validate_conditions_for_theoretical_distns(inference_type: str, **kwargs):
    if inference_type == 'single-mean':
        validate_single_mean_conditions(**kwargs)
    elif inference_type == 'single-proportion':
        validate_single_proportion_conditions(**kwargs)
    elif inference_type == 'two-means':
        validate_two_means_conditions(**kwargs)
    elif inference_type == 'two-proportions':
        validate_two_proportions_conditions(**kwargs)
    else:
        pass


def validate_single_mean_conditions(n: float):
    if n < 30:
        error_msg = ("Conditions for theoretical sampling distributions not met: "
                     f"Sample size n={n:.0f} is less than 30. ")
        TheoreticalConditionNotMetWarning(error_msg)


def validate_single_proportion_conditions(n: float, p: float):
    if n * p < 10 or n * (1 - p) < 10:
        error_msg = ("Conditions for theoretical sampling distributions not met: n*p and n*(1-p) must be greater than "
                     f"10. \nSample values: n*p={n:.0f}*{p:.2f}={n * p:.2f}, and "
                     f"n*(1-p)={n:.0f}*{1 - p:.2f}={n * (1 - p):.2f}")
        TheoreticalConditionNotMetWarning(error_msg)


def validate_two_means_conditions(n1: float, n2: float):
    if n1 < 30 or n2 < 30:
        error_msg = ("Conditions for theoretical sampling distributions not met: For each group n must be greater than "
                     f"30. Sample size n1={n1:.0f}, and sample size n2={n2:.0f}")
        TheoreticalConditionNotMetWarning(error_msg)
    pass


def validate_two_proportions_conditions(x1: Tuple, x2: Tuple):
    n1, p1 = x1
    n2, p2 = x2
    if (n1 * p1 < 10 or n1 * (1 - p1) < 10 or
            n2 * p2 < 10 or n2 * (1 - p2) < 10):
        error_msg = ("Conditions for theoretical sampling distributions not met: For each group n*p and n*(1-p) must be"
                     f" greater than 10. \nSample values: \n\tn1*p1={n1:.0f}*{p1:.2f}={n1 * p1:.2f}, "
                     f"n1*(1-p1)={n1:.0f}*{1 - p1:.2f}={n1 * (1 - p1):.2f}, \n\t"
                     f"n2 * p2={n2:.0f} * {p2:.2f}={n2 * p2:.2f}, "
                     f"n2*(1-p2)={n2:.0f}*{1 - p2:.2f}={n2 * (1 - p2):.2f}"
                     )
        TheoreticalConditionNotMetWarning(error_msg)
    pass
