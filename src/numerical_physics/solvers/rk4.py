import numpy as np


def rk4(f, t_span, y0, dt):
    """
    Solve an ODE using the classical fourth-order
    Runge-Kutta method.

    Parameters
    ----------
    f : callable
        Function defining the ODE:

            dy/dt = f(t, y)

    t_span : tuple
        Start and end time:

            (t_start, t_end)

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
        dt
    )

    y0 = np.asarray(y0, dtype=float)

    y = np.zeros(
        (len(t),) + y0.shape
    )

    y[0] = y0

    for n in range(len(t) - 1):

        tn = t[n]
        yn = y[n]

        # First slope
        k1 = np.asarray(
            f(tn, yn)
        )

        # Second slope
        k2 = np.asarray(
            f(
                tn + dt / 2,
                yn + (dt / 2) * k1
            )
        )

        # Third slope
        k3 = np.asarray(
            f(
                tn + dt / 2,
                yn + (dt / 2) * k2
            )
        )

        # Fourth slope
        k4 = np.asarray(
            f(
                tn + dt,
                yn + dt * k3
            )
        )

        # Weighted average
        y[n + 1] = yn + (
            dt / 6
        ) * (
            k1
            + 2 * k2
            + 2 * k3
            + k4
        )

    return t, y