import numpy as np
import matplotlib.pyplot as plt

from numerical_physics.solvers import rk4
from numerical_physics.mechanics.nonlinear_pendulum import (
    nonlinear_pendulum,
)


# Physical parameters
g = 9.81
length = 0.5


# Initial conditions
theta_0 = np.radians(60.0)
omega_0 = 0.0

initial_state = np.array([
    theta_0,
    omega_0,
])


# Simulation
t, state = rk4(
    lambda t, state: nonlinear_pendulum(
        t,
        state,
        g=g,
        length=length,
    ),
    (0.0, 10.0),
    initial_state,
    0.001,
)


theta = state[:, 0]
omega = state[:, 1]


# Convert angle to degrees for visualization
theta_degrees = np.degrees(theta)


# ------------------------------------------------------------
# Plot angle
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    t,
    theta_degrees,
)

plt.xlabel("Time (s)")
plt.ylabel("Angle (degrees)")
plt.title("Nonlinear Pendulum")

plt.grid(True)
plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# Phase-space plot
# ------------------------------------------------------------

plt.figure(figsize=(7, 7))

plt.plot(
    theta,
    omega,
)

plt.xlabel("Angle θ (rad)")
plt.ylabel("Angular velocity ω (rad/s)")
plt.title("Nonlinear Pendulum Phase Space")

plt.grid(True)
plt.tight_layout()

plt.show()