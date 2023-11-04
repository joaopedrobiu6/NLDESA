import jax.numpy as jnp
import matplotlib.pyplot as plt
from nldesa import EquationSystem

# Define the system of equations
def func(y, t, a):
    """Equation system of an exponencial decay."""
    dydt = -a[0]*y
    return jnp.asarray([dydt])

# Define the initial conditions
y0 = jnp.asarray([1.0])

# Define the time interval
T_0 = 0.0
T_1 = 10.0
N = 101

# Define the parameters
a = jnp.asarray([0.1])

# Create the equation system
eqsys = EquationSystem(func, y0, T_0, T_1, N)

# Solve the equation system
t, y = eqsys.solve(a=a)

# Plot the solution
plot = eqsys.plot_solution(0)
plt.show()
