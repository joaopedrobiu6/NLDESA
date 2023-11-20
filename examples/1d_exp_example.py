import jax.numpy as jnp
import matplotlib.pyplot as plt
from nldesa import EquationSystem
from diffrax import ODETerm

# Define the system of equations
def func(y, t, args):
    """Equation system of an exponencial decay."""
    a0 = args
    dydt = -a0*y
    return jnp.asarray([dydt])

# Define the initial conditions
y0 = jnp.asarray([1.0])

# Define the time interval
T_0 = 0.0
T_1 = 10.0
N = 101

# Define the parameters
args = (0.1)

# Create the equation system
terms = ODETerm(func)
eqsys = EquationSystem(terms, y0, T_0, T_1, N)

# Solve the equation system
t, y = eqsys.solve(a=args)

# Plot the solution
plot = eqsys.plot_solution(0)
plt.show()
