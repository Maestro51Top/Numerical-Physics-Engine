import numpy as np

from numerical_physics.solvers.euler import euler


def test_euler_exponential_decay():
    """Test Euler's method against x(t) = exp(-t)."""

    def decay(t, x):
        return -x

    t, x = euler(
        decay,
        (0.0, 1.0),
        1.0,
        0.001
    )

    analytical = np.exp(-t)

    error = np.max(np.abs(x - analytical))

    assert error < 0.001