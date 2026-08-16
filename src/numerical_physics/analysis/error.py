import numpy as np


def absolute_error(numerical, analytical):
    """
    Calculate the absolute error between numerical
    and analytical solutions.

    Parameters
    ----------
    numerical : array-like
        Numerical solution.

    analytical : array-like
        Analytical solution.

    Returns
    -------
    numpy.ndarray
        Absolute error at every point.
    """

    numerical = np.asarray(numerical)
    analytical = np.asarray(analytical)

    return np.abs(numerical - analytical)


def maximum_error(numerical, analytical):
    """
    Return the maximum absolute error.
    """

    error = absolute_error(numerical, analytical)

    return np.max(error)


def final_error(numerical, analytical):
    """
    Return the absolute error at the final time step.
    """

    error = absolute_error(numerical, analytical)

    return error[-1]