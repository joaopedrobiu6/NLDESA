import jax.numpy as jnp
from jax.experimental.ode import odeint as jax_odeint
from functools import partial
from jax import jit
import matplotlib.pyplot as plt

class Equation_System:
    def __init__(self, f, y0, t0, t1, n):
        self.f = f
        self.y0 = y0
        self.t0 = t0
        self.t1 = t1
        self.n = n
        self.t = jnp.linspace(t0, t1, n)

    def solve(self, a=None, rtol = 1e-10, atol = 1e-10, return_solution = False):
        f_jit = jit(partial(self.f, a=a))
        self.solution = jax_odeint(f_jit, self.y0, self.t, rtol=rtol, atol=atol)
        if return_solution:
            return self.solution
        else:
            return self.t, self.solution
        
    def x(self):
        return self.t
    
    def initial_state(self):
        return self.y0

    def plot_solution(self, component, title = None, xlabel = None, ylabel = None, legend = None):
        fig, ax = plt.subplots()
        if title is not None:
            ax.set_title(title)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if legend is not None:
            ax.legend(legend)
        plot = ax.plot(self.t, self.solution[:, component])
        return plot
    
    