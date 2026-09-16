import numpy as np


def nonlinear_pendulum(t, state, g=9.81, length=1.0):
    """
    Nonlinear simple pendulum.

    State vector:

        state[0] = theta
        state[1] = omega

    Equations:

        dtheta/dt = omega

        domega/dt = -(g/L) * sin(theta)
    """

    theta, omega = state

    dtheta_dt = omega

    domega_dt = -(g / length) * np.sin(theta)

    return np.array([
        dtheta_dt,
        domega_dt,
    ])