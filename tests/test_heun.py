import numpy as np

from numerical_physics.solvers.heun import heun


def test_heun_exponential_growth():

    def growth(t, y):
        return y

    t, numerical = heun(
        growth,
        (0.0, 1.0),
        1.0,
        0.01,
    )

    analytical = np.exp(t)

    error = np.max(
        np.abs(numerical - analytical)
    )

    assert error < 0.0001