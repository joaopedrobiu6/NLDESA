""" pytest for StabilityAnalysis class"""

import numpy as np
import jax.numpy as jnp
import pytest

from nldesa import StabilityAnalysis

class TestStabilityAnalysis:
    """Test the StabilityAnalysis class"""

    @pytest.mark.unit
    def test_stabilityanalysis(self):
        """Test the StabilityAnalysis class"""
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

        eqsys = StabilityAnalysis(system, y0, T_0, T_1, N, a=a)

        eqsys.HODMD(component)
        stab0 = eqsys.stability(component)
        np.testing.assert_allclose(stab0, 1, rtol=1e-5, atol=1e-5)