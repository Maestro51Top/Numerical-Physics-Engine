"""
Numerical Physics Engine

A Python framework for numerical methods,
scientific analysis, and computational physics.
"""

from .solvers import euler, heun

from .analysis import (
    absolute_error,
    maximum_error,
    final_error,
    estimate_convergence_order,
    convergence_table,
)

__all__ = [
    "euler",
    "heun",
    "absolute_error",
    "maximum_error",
    "final_error",
    "estimate_convergence_order",
    "convergence_table",
]