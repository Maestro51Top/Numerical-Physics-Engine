import numpy as np
import matplotlib.pyplot as plt

from numerical_physics.solvers import euler

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

    # Solve using Euler
    t, x_numerical = euler(
        decay,
        (0.0, 5.0),
        1.0,
        dt,
    )

    # Analytical solution
    x_analytical = np.exp(-t)

    # Calculate maximum absolute error
    error = maximum_error(
        x_numerical,
        x_analytical,
    )

    errors.append(error)

    print(
        f"dt = {dt:.6f}, "
        f"maximum error = {error:.8f}"
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
# Print convergence table
# ============================================================

table = convergence_table(
    dt_values,
    errors,
)


print()
print("=" * 60)
print("Euler Method Convergence Analysis")
print("=" * 60)

print(
    f"{'dt':>12}"
    f"{'Error':>20}"
    f"{'Error Ratio':>20}"
)

print("-" * 60)


for row in table:

    dt = row["dt"]
    error = row["error"]
    ratio = row["error_ratio"]

    if np.isnan(ratio):
        ratio_text = "---"
    else:
        ratio_text = f"{ratio:.6f}"

    print(
        f"{dt:>12.6f}"
        f"{error:>20.10e}"
        f"{ratio_text:>20}"
    )


print("-" * 60)

print(
    f"Estimated convergence order: "
    f"{order:.6f}"
)

print("=" * 60)


# ============================================================
# Convergence plot
# ============================================================

plt.figure(figsize=(8, 5))

plt.loglog(
    dt_values,
    errors,
    "o-",
    label="Euler error",
)

plt.xlabel("Time step Δt")
plt.ylabel("Maximum error")
plt.title("Euler Method Convergence")

plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()