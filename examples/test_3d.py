import jax.numpy as jnp
import matplotlib.pyplot as plt
from nldesa import EquationSystem

# Lorentz force equation


def lorentz_force(w, t, a):
    x, y, z, vx, vy, vz = w
    m, e, Ex, Ey, Ez, Bx, By, Bz = a
    dydt = jnp.asarray([vx, vy, vz, -(e/m)*Ex - (e/m)*(vy*Bz-vz*By), -
                       (e/m)*Ey - (e/m)*(vz*Bx - vx*Bz), -(e/m)*Ez - (e/m)*(vx*By-vy*Bx)])
    return dydt


# Define the initial conditions
w0 = jnp.asarray([0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
# Define the time interval
t0 = 0.0
t1 = 30.0
n = 101
# Define the parameters
a = jnp.asarray([1, 1, 0.1, 0.1, 0, 0, 0, 1])  # m, e, Ex, Ey, Ez, Bx, By, Bz

# Create the equation system
eqsys = EquationSystem(lorentz_force, w0, t0, t1, n)

# Solve the equation system
t, w = eqsys.solve(a=a)

# Plot the phase space
plot = eqsys.plot_phase(jnp.asarray(
    [0, 1, 2]), title='Lorentz force', xlabel='x', ylabel='y', zlabel='z', color='r')
plt.show()
