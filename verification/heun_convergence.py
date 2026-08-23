import numpy as np
import matplotlib.pyplot as plt

from numerical_physics.solvers import heun

from numerical_physics.analysis import (
    maximum_error,
    estimate_convergence_order,
    convergence_table,
)


# ============================================================
# Physical problem
# ============================================================

def decay(t, x):
    """
    Exponential decay equation:

        dx/dt = -x

    Analytical solution:

        x(t) = exp(-t)

    Initial condition:

        x(0) = 1
    """
    return -x


# ============================================================
# Numerical experiment
# ============================================================

dt_values = np.array([
    0.5,
    0.25,
    0.125,
    0.0625,
    0.03125,
    0.015625,
])

errors = []


for dt in dt_values:

    # Solve using Heun
    t, x_numerical = heun(
        decay,
        (0.0, 5.0),
        1.0,
        dt,
    )

    # Analytical solution
    x_analytical = np.exp(-t)

    # Maximum absolute error
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


# ============================================================
# Estimate convergence order
# ============================================================

order = estimate_convergence_order(
    dt_values,
    errors,
)


# ============================================================
# Generate convergence table
# ============================================================

table = convergence_table(
    dt_values,
    errors,
)


print()
print("=" * 65)
print("Heun Method Convergence Analysis")
print("=" * 65)

print(
    f"{'dt':>12}"
    f"{'Maximum Error':>22}"
    f"{'Error Ratio':>20}"
    f"{'Observed Order':>20}"
)

print("-" * 65)


for row in table:

    dt = row["dt"]
    error = row["error"]
    ratio = row["error_ratio"]
    order_value = row["observed_order"]

    if np.isnan(ratio):
        ratio_text = "---"
    else:
        ratio_text = f"{ratio:.6f}"

    if np.isnan(order_value):
        order_text = "---"
    else:
        order_text = f"{order_value:.6f}"

    print(
        f"{dt:>12.6f}"
        f"{error:>22.10e}"
        f"{ratio_text:>20}"
        f"{order_text:>20}"
    )


print("-" * 65)

print(
    f"Estimated convergence order: "
    f"{order:.6f}"
)

print("=" * 65)


# ============================================================
# Convergence plot
# ============================================================

plt.figure(figsize=(8, 5))

plt.loglog(
    dt_values,
    errors,
    "o-",
    label="Heun error",
)

plt.xlabel("Time step Δt")
plt.ylabel("Maximum error")
plt.title("Heun Method Convergence")

plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()