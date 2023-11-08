import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import time as time

from nldesa import StabilityMap

init_time = time.time()
# Define the system
def system(y, t, a):
    """Equation system of a pendulum."""
    theta, omega = y
    dydt = jnp.asarray([omega, -(a[0] + a[1]*jnp.cos(t))*jnp.sin(theta)])
    return dydt

# Define the initial conditions
y0 = jnp.asarray([jnp.pi - 0.1, 0.0])

# Define the time interval
T_0 = 0.0
T_1 = 100.0
N = 100
component = 0
# Define the parameters

a0 = jnp.linspace(0, 0.1, 10)
a1 = jnp.linspace(0, 0.1, 10)
a = jnp.asarray([a0, a1])
# First column are the values of a[0], second column are the values of a[1]

intermediate_time = time.time()
print(f"Time to define the system: {intermediate_time - init_time}")
status = StabilityMap(system, y0, T_0, T_1, N, a, component)

intermediate_time2 = time.time()
print(f"Time to solve the system: {intermediate_time2 - intermediate_time}")
print(status[:, 2])
final_time = time.time()
print(f"Total time: {final_time - init_time}")