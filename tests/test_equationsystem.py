""" pytest for eqsystem.py"""

import jax
import jax.numpy as jnp
import pytest

from nldesa import EquationSystem

class TestEquationSystem:
    """Test the EquationSystem class"""

    @pytest.mark.unit
    def test_equationsystem(self):
        """Test the EquationSystem class"""
        # Define the differential equation system
        def f(y, t, a):
            return jnp.asarray([a[0]*y[0] - a[1]*y[0]*y[1], -a[2]*y[1] + a[3]*y[0]*y[1]])

        # Define the initial conditions
        y0 = jnp.asarray([1.0, 1.0])
        # Define the time interval
        t0 = 0.0
        t1 = 5.0
        n = 101
        # Define the parameters
        a = jnp.asarray([1, 1, 1, 1])

        # Create the equation system
        eqsys = EquationSystem(f, y0, t0, t1, n)

        # Solve the equation system
        t, y = eqsys.solve(a=a)

        # Test the solution
        assert jnp.allclose(y[0], jnp.asarray([1.0, 1.0, 1.0, 1.0, 1.0]))
        assert jnp.allclose(y[1], jnp.asarray([1.0, 0.99999994, 0.99999994, 0.99999994, 0.99999994]))
        assert jnp.allclose(t, jnp.asarray([0.0, 0.05, 0.1, 0.15, 0.2]))
