""" Test for the EquationSystem class using the pendulum equation in 1D. """
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pydmd import HODMD
from pydmd.plotter import plot_eigs
from nldesa import EquationSystem


# Define the system of equations
def pendulum(y, t, a):
    """Equation system of the pendulum."""
    theta, omega = y
    dydt = jnp.asarray([omega, (a[0] + a[1]*jnp.cos(t))*jnp.sin(theta)])
    return dydt


# Define the initial conditions
y0 = jnp.asarray([jnp.pi - 0.1, 0.0])

# Define the time interval
T_0 = 0.0
T_1 = 30.0
N = 101

# Define the parameters
a = jnp.asarray([0.1, 0.1])
a2 = jnp.asarray([0.8, 0.1])


# Create the equation system
eqsys = EquationSystem(pendulum, y0, T_0, T_1, N)

# Solve the equation system
t, sol = eqsys.solve(a=a)
t2, sol2 = eqsys.solve(a=a2)

# Plot the solution
# plot = eqsys.plot_solution(0, title='Pendulum', xlabel='t', ylabel='theta')
# plt.show()

# plt.plot(t, sol[:, 0])
# plt.show()


x = t
y = sol[:, 0]
y2 = sol2[:, 0]

hodmd = HODMD(svd_rank=0, exact=True, opt=True, d=30).fit(y[None])
hodmd2 = HODMD(svd_rank=0, exact=True, opt=True, d=30).fit(y2[None])

print(jnp.mean(jnp.sqrt(hodmd.eigs.real**2 + hodmd.eigs.imag**2)))
print(jnp.mean(jnp.sqrt(hodmd2.eigs.real**2 + hodmd2.eigs.imag**2)))
print(jnp.mean(hodmd.eigs.real**2 + hodmd.eigs.imag**2)**2 - jnp.mean(hodmd2.eigs.real**2 + hodmd2.eigs.imag**2)**2)