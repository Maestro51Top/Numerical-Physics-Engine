import numpy as np

from numerical_physics.solvers import rk4


def test_rk4_exponential_decay():

    def decay(t, x):
        return -x

    t, x = rk4(
        decay,
        (0.0, 5.0),
        1.0,
        0.1,
    )

    analytical = np.exp(-t)

    error = np.max(
        np.abs(x - analytical)
    )

    assert error < 1e-5