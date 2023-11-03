""" Test for the EquationSystem class using the pendulum equation in 1D. """
import jax.numpy as jnp
import matplotlib.pyplot as plt
from nldesa import EquationSystem


# Define the system of equations
def pendulum(y_val, _, a_val):
    """Equation system of the pendulum."""
    theta, omega = y_val
    dydt = jnp.asarray([omega, a_val[0] + a_val[1]*jnp.sin(theta)])
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
eqsys = EquationSystem(pendulum, y0, T_0, T_1, N)

# Solve the equation system
t, y = eqsys.solve(a=a)

# Plot the solution
plot = eqsys.plot_solution(0, title='Pendulum', xlabel='t', ylabel='theta')
plt.show()
