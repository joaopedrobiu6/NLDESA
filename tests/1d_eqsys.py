from nldesa import Equation_System
import jax.numpy as jnp
import matplotlib.pyplot as plt


# Define the system of equations
def pendulum(y, t, a):
    theta, omega = y
    dydt = jnp.asarray([omega, a[0] + a[1]*jnp.sin(theta)])
    return dydt

# Define the initial conditions
y0 = jnp.asarray([jnp.pi - 0.1, 0.0])

# Define the time interval
t0 = 0.0
t1 = 10.0
n = 101

# Define the parameters
a = jnp.asarray([0.0, -9.81])

# Create the equation system
eqsys = Equation_System(pendulum, y0, t0, t1, n)

# Solve the equation system
t, y = eqsys.solve(a = a)

# Plot the solution
plot = eqsys.plot_solution(0, title = 'Pendulum', xlabel = 't', ylabel = 'theta')
plt.show()
