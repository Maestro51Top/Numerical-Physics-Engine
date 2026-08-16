import numpy as np
import matplotlib.pyplot as plt

from numerical_physics.solvers.euler import euler
from numerical_physics.analysis import (
    maximum_error,
    final_error,
)


# ============================================================
# Physical parameters
# ============================================================

mass = 1.0          # kg
spring_constant = 1.0  # N/m

omega = np.sqrt(spring_constant / mass)

# Initial conditions
x0 = 1.0            # m
v0 = 0.0            # m/s

dt = 0.001           # s
t_span = (0.0, 10.0)


# ============================================================
# Harmonic oscillator differential equation
# ============================================================

def harmonic_oscillator(t, y):
    """
    Harmonic oscillator:

        dx/dt = v
        dv/dt = -omega^2 * x

    State vector:

        y = [x, v]
    """

    x, v = y

    dxdt = v
    dvdt = -(omega ** 2) * x

    return np.array([dxdt, dvdt])


# ============================================================
# Numerical solution
# ============================================================

t, numerical = euler(
    harmonic_oscillator,
    t_span,
    np.array([x0, v0]),
    dt
)

x_numerical = numerical[:, 0]
v_numerical = numerical[:, 1]


# ============================================================
# Analytical solution
# ============================================================

x_analytical = x0 * np.cos(omega * t)

v_analytical = -x0 * omega * np.sin(omega * t)


# ============================================================
# Error analysis
# ============================================================

position_max_error = maximum_error(
    x_numerical,
    x_analytical
)

position_final_error = final_error(
    x_numerical,
    x_analytical
)


print("==========================================")
print("Euler Harmonic Oscillator")
print("==========================================")

print(f"Mass                 = {mass} kg")
print(f"Spring constant      = {spring_constant} N/m")
print(f"Angular frequency    = {omega:.4f} rad/s")
print(f"Time step            = {dt} s")

print()
print(f"Maximum position error = {position_max_error:.10e}")
print(f"Final position error   = {position_final_error:.10e}")


# ============================================================
# Energy calculation
# ============================================================

kinetic_energy = 0.5 * mass * v_numerical**2

potential_energy = (
    0.5 * spring_constant * x_numerical**2
)

total_energy = kinetic_energy + potential_energy


# Analytical initial energy
initial_energy = (
    0.5 * mass * v0**2
    + 0.5 * spring_constant * x0**2
)


energy_error = np.abs(total_energy - initial_energy)

maximum_energy_error = np.max(energy_error)


print()
print(f"Initial energy          = {initial_energy:.10e} J")
print(f"Final numerical energy  = {total_energy[-1]:.10e} J")
print(f"Maximum energy error    = {maximum_energy_error:.10e} J")


# ============================================================
# Plot 1 — Position
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(
    t,
    x_numerical,
    label="Euler"
)

plt.plot(
    t,
    x_analytical,
    label="Analytical"
)

plt.xlabel("Time (s)")
plt.ylabel("Position x (m)")
plt.title("Euler Method — Harmonic Oscillator")

plt.grid(True)
plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# Plot 2 — Phase space
# ============================================================

plt.figure(figsize=(7, 7))

plt.plot(
    x_numerical,
    v_numerical,
    label="Euler"
)

plt.xlabel("Position x (m)")
plt.ylabel("Velocity v (m/s)")
plt.title("Phase Space — Harmonic Oscillator")

plt.grid(True)
plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# Plot 3 — Energy
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(
    t,
    total_energy,
    label="Euler numerical energy"
)

plt.axhline(
    initial_energy,
    linestyle="--",
    label="Analytical energy"
)

plt.xlabel("Time (s)")
plt.ylabel("Energy (J)")
plt.title("Energy Conservation — Euler Method")

plt.grid(True)
plt.legend()

plt.tight_layout()

plt.show()