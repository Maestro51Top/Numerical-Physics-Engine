"""
Heun's method for solving first-order ordinary differential equations.
"""

import numpy as np


def heun(f, t_span, y0, dt):
    """
    Solve a first-order ODE using Heun's method.

    The equation is:

        dy/dt = f(t, y)

    Parameters
    ----------
    f : callable
        Function defining the differential equation.

    t_span : tuple
        Start and end time, (t_start, t_end).

    y0 : float or array-like
        Initial condition.

    dt : float
        Time step.

    Returns
    -------
    t : numpy.ndarray
        Time values.

    y : numpy.ndarray
        Numerical solution.
    """

    t_start, t_end = t_span

    t = np.arange(
        t_start,
        t_end + dt,
        dt,
    )

    y0 = np.asarray(
        y0,
        dtype=float,
    )

    y = np.zeros(
        (len(t),) + y0.shape
    )

    y[0] = y0

    for n in range(len(t) - 1):

        # Predictor
        k1 = np.asarray(
            f(t[n], y[n])
        )

        y_predict = (
            y[n] + dt * k1
        )

        # Corrector
        k2 = np.asarray(
            f(t[n + 1], y_predict)
        )

        y[n + 1] = (
            y[n]
            + (dt / 2.0) * (k1 + k2)
        )

    return t, y