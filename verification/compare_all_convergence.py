import numpy as np
import matplotlib.pyplot as plt

from numerical_physics.solvers import euler, heun, rk4
from numerical_physics.analysis import (
    maximum_error,
    estimate_convergence_order,
)


def decay(t, x):
    """dx/dt = -x"""
    return -x


# Same experiment for every solver
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
rk4_errors = []


for dt in dt_values:

    # -------------------------
    # Euler
    # -------------------------

    t, x = euler(
        decay,
        (0.0, 5.0),
        1.0,
        dt,
    )

    exact = np.exp(-t)

    euler_errors.append(
        maximum_error(x, exact)
    )


    # -------------------------
    # Heun
    # -------------------------

    t, x = heun(
        decay,
        (0.0, 5.0),
        1.0,
        dt,
    )

    exact = np.exp(-t)

    heun_errors.append(
        maximum_error(x, exact)
    )


    # -------------------------
    # RK4
    # -------------------------

    t, x = rk4(
        decay,
        (0.0, 5.0),
        1.0,
        dt,
    )

    exact = np.exp(-t)

    rk4_errors.append(
        maximum_error(x, exact)
    )


euler_errors = np.array(euler_errors)
heun_errors = np.array(heun_errors)
rk4_errors = np.array(rk4_errors)


# ------------------------------------------------------------
# Convergence orders
# ------------------------------------------------------------

euler_order = estimate_convergence_order(
    dt_values,
    euler_errors,
)

heun_order = estimate_convergence_order(
    dt_values,
    heun_errors,
)

rk4_order = estimate_convergence_order(
    dt_values,
    rk4_errors,
)


# ------------------------------------------------------------
# Print results
# ------------------------------------------------------------

print()
print("=" * 100)
print("Euler vs Heun vs RK4 Convergence Analysis")
print("=" * 100)

print(
    f"{'dt':>12}"
    f"{'Euler Error':>22}"
    f"{'Heun Error':>22}"
    f"{'RK4 Error':>22}"
)

print("-" * 100)


for dt, euler_error, heun_error, rk4_error in zip(
    dt_values,
    euler_errors,
    heun_errors,
    rk4_errors,
):

    print(
        f"{dt:>12.6f}"
        f"{euler_error:>22.10e}"
        f"{heun_error:>22.10e}"
        f"{rk4_error:>22.10e}"
    )


print("-" * 100)

print(
    f"Euler observed order: {euler_order:.6f}"
)

print(
    f"Heun observed order:  {heun_order:.6f}"
)

print(
    f"RK4 observed order:   {rk4_order:.6f}"
)

print("=" * 100)


# ------------------------------------------------------------
# Unified convergence plot
# ------------------------------------------------------------

plt.figure(figsize=(9, 6))

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

plt.loglog(
    dt_values,
    rk4_errors,
    "^-",
    label=f"RK4 (p ≈ {rk4_order:.2f})",
)

plt.xlabel("Time step Δt")
plt.ylabel("Maximum error")
plt.title("Numerical Solver Convergence Comparison")

plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()