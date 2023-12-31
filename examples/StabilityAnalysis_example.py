import jax.numpy as jnp
import matplotlib.pyplot as plt
from nldesa import StabilityAnalysis

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
N = 1001
component = 0
# Define the parameters
a = jnp.asarray([0.0, 0.001])
print(f"Params: {a}")

eqsys = StabilityAnalysis(system, y0, T_0, T_1, N, a=a)

eqsys.HODMD(component)

eig = eqsys.eigenvalues(component)
print(eig)

eqsys.plot_eigenvalues(component)
plt.show()

print(eqsys.stability(component))