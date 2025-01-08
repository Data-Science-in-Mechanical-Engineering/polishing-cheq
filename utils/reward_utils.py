import numpy as np

def _compute_penalty(
        value: float, 
        target_value: float, 
        lower_limit: float, 
        upper_limit:float
    ) -> float:
    """ 
    Computes the deviation of value and target_value in percent.
    Given an lower_limit and an upper_limit.

    lower_limit --------- target ----x---- upper_limit --> x is the current value.

    Deviations between lower_limit and target is scaled to [0,1] 
    Deviations between upper_limit and target is scaled to [0,1] 
    Deviations above or below the limits are 1.

    Returns: 
        float: Relative penalty [0,1] clipped to boundaries
    """

    penalty = 0
    delta_value = np.abs(value - target_value)

    if value > target_value:  # we are exceeding the target value
        penalty = delta_value / np.abs(upper_limit - target_value)

    if value < target_value:  # we are below the target value
        penalty = delta_value / np.abs(target_value - lower_limit)

    if penalty > 1:  # If outside [lower_limit, upper_limit] give maximum penalty
        penalty = 1

    return penalty

def _compute_linear_reward(
        value: float, 
        target_value: float, 
        lower_limit: float, 
        upper_limit: float
    ) -> float:
    """
    Computes the counter function of the _compute_penalts() method. \
    All rewards are bounded between [0, 1]. \
    If the value is close to the target value, will lead to high rewards. 

    Returns:
        float: Relative reward bounded between [0, 1]
    """
    return 1 - _compute_penalty(value, target_value, lower_limit, upper_limit)

def _compute_gaussian_reward(value: float, theta: np.ndarray) -> float:
    """
    Computes the reward based on an auxiliary function similar to the gaussian curve.

    g(x, theta) = theta[0] * exp(-(x^2)/(2*theta[1]^2)) 

    Returns:
        float: The reward based on the target and theta function. Bounded between [0, theta_0].
    """ 
    return theta[0] * np.exp( - (value)**2 / (2 * theta[1]**2)) 

def _compute_piecewise_linear_quad_reward(
        value: float,
        target_value: float,
        lower_limit: float,
        lower_limit_inner: float,
        upper_limit_inner: float,
        upper_limit: float,
    ) -> float:
    """
    Computes the piecewise linear and quadratic reward used for the force of the real robot.
    In the inner bounds it will compute a linear reward function value 
    while the outer bounds are used for the quadratic reward function.
    All function values are bounded by [0,1].

    Parts of the function: 0 --- quadratic term --- linear term --- quadratic term --- 0

    Returns:
        float: Relative reward bounded between [0,1]
    """
    assert lower_limit < lower_limit_inner < upper_limit_inner < upper_limit, "Please make sure to provide the correct bounds."

    objective = value - target_value
    if (objective > lower_limit_inner) and (objective < upper_limit_inner):
        # linear part
        if objective > 0:
            result = 0.5/(lower_limit_inner)*objective + 1
        else:
            result = 0.5/(upper_limit_inner)*objective + 1
    elif (objective > lower_limit) and (objective < upper_limit):
        # outer quadratic part
        a = 0.5 / (lower_limit_inner**2 - lower_limit**2)
        c = -(0.5*lower_limit**2) / (lower_limit_inner**2 - lower_limit**2)
        result = a * objective**2 + c
    else:
        # completely out of bounds
        result = 0
    assert 0 <= result <= 1, f"Reward value {result} for value {value} out of [0,1]... this should not be the case"
    return result


def _normalize_weighting(*args):
    """
    Normalizes all weighting coefficients given as arguments to the sum of 1.
    """
    sum_coefs = 0.0
    for arg in args:
        assert isinstance(arg, float), "Assumed to receive floats as arguments."
        sum_coefs += arg

    norm_args = (arg/sum_coefs for arg in args)
    return norm_args