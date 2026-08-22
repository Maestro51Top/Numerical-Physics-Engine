"""
Convergence analysis utilities for numerical methods.
"""

import numpy as np


def estimate_convergence_order(dt_values, errors):
    """
    Estimate the numerical convergence order.

    Parameters
    ----------
    dt_values : array-like
        Time-step sizes.

    errors : array-like
        Corresponding numerical errors.

    Returns
    -------
    float
        Estimated convergence order.

    Notes
    -----
    If

        error ~ C * dt^p

    then

        log(error) = log(C) + p * log(dt)

    Therefore, p can be estimated from the slope
    of log(error) versus log(dt).
    """

    dt_values = np.asarray(dt_values, dtype=float)
    errors = np.asarray(errors, dtype=float)

    if len(dt_values) != len(errors):
        raise ValueError("dt_values and errors must have the same length.")

    if len(dt_values) < 2:
        raise ValueError("At least two data points are required.")

    if np.any(dt_values <= 0):
        raise ValueError("Time-step values must be positive.")

    if np.any(errors <= 0):
        raise ValueError("Errors must be positive.")

    log_dt = np.log(dt_values)
    log_error = np.log(errors)

    order, _ = np.polyfit(log_dt, log_error, 1)

    return float(order)


def convergence_table(dt_values, errors):
    """
    Create a simple convergence table.

    Returns
    -------
    list of dict
        Each row contains dt, error, and error ratio.
    """

    dt_values = np.asarray(dt_values, dtype=float)
    errors = np.asarray(errors, dtype=float)

    if len(dt_values) != len(errors):
        raise ValueError("dt_values and errors must have the same length.")

    table = []

    for i, (dt, error) in enumerate(zip(dt_values, errors)):

        if i == 0:
            ratio = np.nan
        else:
            ratio = errors[i - 1] / error

        table.append(
            {
                "dt": dt,
                "error": error,
                "error_ratio": ratio,
            }
        )

    return table