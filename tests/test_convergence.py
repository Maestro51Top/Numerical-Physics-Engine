import numpy as np

from numerical_physics.analysis.convergence import (
    estimate_convergence_order,
    convergence_table,
)


def test_euler_like_first_order_convergence():
    dt_values = np.array([
        0.1,
        0.05,
        0.025,
        0.0125,
    ])

    errors = np.array([
        0.1,
        0.05,
        0.025,
        0.0125,
    ])

    order = estimate_convergence_order(dt_values, errors)

    assert np.isclose(order, 1.0)


def test_convergence_table():

    dt_values = np.array([
        0.1,
        0.05,
        0.025,
    ])

    errors = np.array([
        0.1,
        0.05,
        0.025,
    ])

    table = convergence_table(dt_values, errors)

    assert len(table) == 3

    assert np.isnan(table[0]["error_ratio"])

    assert np.isclose(table[1]["error_ratio"], 2.0)

    assert np.isclose(table[2]["error_ratio"], 2.0)