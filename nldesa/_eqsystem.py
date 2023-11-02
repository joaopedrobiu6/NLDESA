import jax.numpy as jnp
from jax.experimental.ode import odeint as jax_odeint

class Equation_System:
    def __init__(self, f, y0, t0, t1, n):
        self.f = f
        self.y0 = y0
        self.t0 = t0
        self.t1 = t1
        self.n = n
        self.t = jnp.linspace(t0, t1, n)

    def solve(self, rtol = 1e-10, atol = 1e-10, return_solution = False):
        self.solution = jax_odeint(self.f, self.y0, self.t, rtol=rtol, atol=atol)
        if return_solution:
            return self.solution
        else:
            return self.t, self.solution
        
    def x(self):
        return self.t
    
    def initial_state(self):
        return self.y0