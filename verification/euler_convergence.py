import numpy as np
import matplotlib.pyplot as plt

from numerical_physics.solvers.euler import euler


def decay(t, x):
    return -x


# Different time steps
dt_values = [
    0.5,
    0.25,
    0.125,
    0.0625,
    0.03125,
    0.015625,
]


errors = []


for dt in dt_values:

    t, x = euler(
        decay,
        (0.0, 5.0),
        1.0,
        dt
    )

    analytical = np.exp(-t)

    error = np.max(np.abs(x - analytical))

    errors.append(error)

    print(f"dt = {dt:.6f}, error = {error:.8f}")


# Plot convergence
plt.figure(figsize=(8, 5))

plt.loglog(
    dt_values,
    errors,
    "o-",
    label="Euler error"
)

plt.xlabel("Time step Δt")
plt.ylabel("Maximum error")
plt.title("Euler Method Convergence")

plt.grid(True)
plt.legend()

plt.show()