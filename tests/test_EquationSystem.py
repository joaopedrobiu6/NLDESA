""" pytest for EquationSystem class"""

import numpy as np
import jax.numpy as jnp
import pytest

from nldesa import EquationSystem

class TestEquationSystem:
    """Test the EquationSystem class"""

    @pytest.mark.unit
    def test_equationsystem(self):
        """Test the EquationSystem class"""
        # Define the differential equation system
        def exponential(y, t, a):
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
        eqsys = EquationSystem(exponential, y0, T_0, T_1, N)

        # Solve the equation system
        t, y = eqsys.solve(a=a)

        # Check the solution
        np.testing.assert_allclose(y[:, 0], jnp.exp(-a[0]*t), rtol=1e-5, atol=1e-5)