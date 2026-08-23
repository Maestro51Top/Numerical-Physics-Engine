import numpy as np
import matplotlib.pyplot as plt

from numerical_physics.solvers import euler, heun
from numerical_physics.analysis import (
    maximum_error,
    estimate_convergence_order,
)


# ============================================================
# Physical problem
# ============================================================

def decay(t, x):
    """
    Exponential decay:

        dx/dt = -x

    Exact solution:

        x(t) = exp(-t)

    Initial condition:

        x(0) = 1
    """
    return -x


# ============================================================
# Experimental setup
# ============================================================

dt_values = np.array([
    0.5,
    0.25,
    0.125,
    0.0625,
    0.03125,
    0.015625,
])


euler_errors = []
heun_errors = []


# ============================================================
# Run both solvers
# ============================================================

for dt in dt_values:

    # -------------------------
    # Euler
    # -------------------------

    t_euler, x_euler = euler(
        decay,
        (0.0, 5.0),
        1.0,
        dt,
    )

    exact_euler = np.exp(-t_euler)

    euler_error = maximum_error(
        x_euler,
        exact_euler,
    )

    euler_errors.append(euler_error)

    # -------------------------
    # Heun
    # -------------------------

    t_heun, x_heun = heun(
        decay,
        (0.0, 5.0),
        1.0,
        dt,
    )

    exact_heun = np.exp(-t_heun)

    heun_error = maximum_error(
        x_heun,
        exact_heun,
    )

    heun_errors.append(heun_error)


euler_errors = np.array(euler_errors)
heun_errors = np.array(heun_errors)


# ============================================================
# Estimate convergence orders
# ============================================================

euler_order = estimate_convergence_order(
    dt_values,
    euler_errors,
)

heun_order = estimate_convergence_order(
    dt_values,
    heun_errors,
)


# ============================================================
# Print results
# ============================================================

print()
print("=" * 80)
print("Euler vs Heun Convergence Comparison")
print("=" * 80)

print(
    f"{'dt':>12}"
    f"{'Euler Error':>22}"
    f"{'Heun Error':>22}"
    f"{'Euler/Heun':>18}"
)

print("-" * 80)


for dt, e_error, h_error in zip(
    dt_values,
    euler_errors,
    heun_errors,
):

    improvement = e_error / h_error

    print(
        f"{dt:>12.6f}"
        f"{e_error:>22.10e}"
        f"{h_error:>22.10e}"
        f"{improvement:>18.4f}"
    )


print("-" * 80)

print(
    f"Euler observed convergence order: "
    f"{euler_order:.6f}"
)

print(
    f"Heun observed convergence order:  "
    f"{heun_order:.6f}"
)

print("=" * 80)


# ============================================================
# Convergence plot
# ============================================================

plt.figure(figsize=(8, 5))

plt.loglog(
    dt_values,
    euler_errors,
    "o-",
    label=f"Euler (p ≈ {euler_order:.2f})",
)

plt.loglog(
    dt_values,
    heun_errors,
    "s-",
    label=f"Heun (p ≈ {heun_order:.2f})",
)

plt.xlabel("Time step Δt")
plt.ylabel("Maximum error")
plt.title("Euler vs Heun Convergence")

plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()