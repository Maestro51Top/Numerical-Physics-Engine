import numpy as np


def euler(f, t_span, y0, dt):
    """
    Solve a first-order ordinary differential equation
    using Euler's method.

    Parameters
    ----------
    f : callable
        Function representing dy/dt = f(t, y).

    t_span : tuple
        Start and end time (t_start, t_end).

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

    t = np.arange(t_start, t_end + dt, dt)

    y0 = np.asarray(y0, dtype=float)

    y = np.zeros((len(t),) + y0.shape)

    y[0] = y0

    for n in range(len(t) - 1):
        y[n + 1] = y[n] + dt * np.asarray(f(t[n], y[n]))

    return t, y