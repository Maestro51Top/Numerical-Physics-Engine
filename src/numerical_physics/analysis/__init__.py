from .error import (
    absolute_error,
    maximum_error,
    final_error,
)

from .convergence import (
    estimate_convergence_order,
    convergence_table,
)

__all__ = [
    "absolute_error",
    "maximum_error",
    "final_error",
]