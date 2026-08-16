import numpy as np
import matplotlib.pyplot as plt

from numerical_physics.solvers.euler import euler


def decay(t, x):
    return -x


# Simulation parameters
t_start = 0.0
t_end = 5.0
dt = 0.01
x0 = 1.0

# Numerical solution
t, x = euler(decay, (t_start, t_end), x0, dt)

# Analytical solution
x_exact = np.exp(-t)

error = np.abs(x - x_exact)

print("Maximum error:", np.max(error))
print("Final error:", error[-1])
# Plot
plt.figure(figsize=(8, 5))

plt.plot(t, x, "o-", label="Euler")
plt.plot(t, x_exact, label="Analytical")

plt.xlabel("Time")
plt.ylabel("x(t)")
plt.title("Euler Method vs Analytical Solution")
plt.legend()
plt.grid(True)

plt.show()