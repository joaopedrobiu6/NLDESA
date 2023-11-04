import jax.numpy as jnp
import matplotlib.pyplot as plt
from nldesa import StabilityAnalysis

# Define the system
def system(y, t, a):
    """Equation system of a pendulum."""
    theta, omega = y
    dydt = jnp.asarray([omega, a[0] + a[1]*jnp.sin(theta)])
    return dydt

# Define the initial conditions
y0 = jnp.asarray([jnp.pi - 0.1, 0.0])

# Define the time interval
T_0 = 0.0
T_1 = 10.0
N = 101

# Define the parameters
a = jnp.asarray([0.0, -10])

# Create the equation system
eqsys = StabilityAnalysis(system, y0, T_0, T_1, N)

# Solve the equation system
time, solution = eqsys.solve(a=a)

# Plot the solution
plot = eqsys.plot_solution(0, title='Pendulum', xlabel='t', ylabel='theta')
plt.show()