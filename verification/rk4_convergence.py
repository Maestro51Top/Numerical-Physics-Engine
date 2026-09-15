import numpy as np
import matplotlib.pyplot as plt

from numerical_physics.solvers import rk4
from numerical_physics.analysis import (
    maximum_error,
    estimate_convergence_order,
    convergence_table,
)


def decay(t, x):
    """
    Exponential decay:

        dx/dt = -x

    Exact solution:

        x(t) = exp(-t)
    """
    return -x


# ------------------------------------------------------------
# Time-step values
# ------------------------------------------------------------

dt_values = np.array([
    0.5,
    0.25,
    0.125,
    0.0625,
    0.03125,
    0.015625,
])


errors = []


# ------------------------------------------------------------
# Run RK4 for each timestep
# ------------------------------------------------------------

for dt in dt_values:

    t, x_numerical = rk4(
        decay,
        (0.0, 5.0),
        1.0,
        dt,
    )

    x_analytical = np.exp(-t)

    error = maximum_error(
        x_numerical,
        x_analytical,
    )

    errors.append(error)

    print(
        f"dt = {dt:.6f}, "
        f"maximum error = {error:.10e}"
    )


errors = np.array(errors)


# ------------------------------------------------------------
# Convergence order
# ------------------------------------------------------------

order = estimate_convergence_order(
    dt_values,
    errors,
)


# ------------------------------------------------------------
# Convergence table
# ------------------------------------------------------------

table = convergence_table(
    dt_values,
    errors,
)


print()
print("=" * 85)
print("RK4 Method Convergence Analysis")
print("=" * 85)

print(
    f"{'dt':>12}"
    f"{'Maximum Error':>22}"
    f"{'Error Ratio':>20}"
    f"{'Observed Order':>20}"
)

print("-" * 85)


for row in table:

    dt = row["dt"]
    error = row["error"]
    ratio = row["error_ratio"]
    observed_order = row["observed_order"]

    if np.isnan(ratio):
        ratio_text = "---"
    else:
        ratio_text = f"{ratio:.6f}"

    if np.isnan(observed_order):
        order_text = "---"
    else:
        order_text = f"{observed_order:.6f}"

    print(
        f"{dt:>12.6f}"
        f"{error:>22.10e}"
        f"{ratio_text:>20}"
        f"{order_text:>20}"
    )


print("-" * 85)

print(
    f"Estimated convergence order: "
    f"{order:.6f}"
)

print("=" * 85)


# ------------------------------------------------------------
# Convergence plot
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.loglog(
    dt_values,
    errors,
    "o-",
    label=f"RK4 (p ≈ {order:.2f})",
)

plt.xlabel("Time step Δt")
plt.ylabel("Maximum error")

plt.title("RK4 Method Convergence")

plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()