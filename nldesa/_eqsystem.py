from functools import partial
import jax.numpy as jnp
from jax.experimental.ode import odeint as jax_odeint
from jax import jit
import matplotlib.pyplot as plt


class EquationSystem:
    """ Differential Equation System 

    Parameters
    ----------
    f : function
        The function defining the differential equation system. 
        The function must have the signature f(y, t, a), where y is the state vector, t is the time, and a is a parameter vector.
    y0 : array
        The initial state vector.
    t0 : float
        The initial time.
    t1 : float
        The final time.
    n : int
        The number of time steps.
    """

    def __init__(self, f, y0, t0, t1, n):
        self.f = f
        self.y0 = y0
        self.t0 = t0
        self.t1 = t1
        self.n = n
        self.t = jnp.linspace(t0, t1, n)

    def solve(self, a=None, rtol=1e-10, atol=1e-10, return_solution=False):
        """ Solve the equation system.

        Parameters
        ----------
        a : array
            The parameter vector.
        rtol : float
            The relative tolerance.
        atol : float
            The absolute tolerance.

        Returns
        -------
        t : array
            The time values.
        solution : array
            The solution.
        """

        f_jit = jit(self.f)
        self.solution = jax_odeint(partial(f_jit, a=a), self.y0, self.t, rtol=rtol, atol=atol)
        # self.solution = jax_odeint(
        #     f_jit, self.y0, self.t, rtol=rtol, atol=atol)
        if return_solution:
            return self.solution
        else:
            return self.t, self.solution

    def x(self):
        """ Return the time value."""
        return self.t

    def initial_state(self):
        """ Return the initial state vector."""
        return self.y0

    def plot_solution(self, component, title=None, xlabel=None, ylabel=None, legend=None):
        """ Plot the solution.

        Parameters
        ----------
        component : int
            The component of the state vector to plot. 0 is the function value, 1 is the 1st derivative, etc.
        title : string
            The title of the plot.
        xlabel : string
        The label of the x-axis.
        ylabel : string
            The label of the y-axis.
        legend : string
            The legend of the plot.

        Returns
        -------
        plot : matplotlib plot
            The plot.
        """

        ax = plt.figure().add_subplot()
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

    def plot_phase(self, components, title=None, xlabel=None, ylabel=None, zlabel=None, legend=None, **kwargs):
        """ Plot the phase space.

        Parameters
        ----------
        components : array
            The components of the state vector to plot. 0 is the function value, 1 is the 1st derivative, etc.
        title : string
            The title of the plot.
        xlabel : string
            The label of the x-axis.
        ylabel : string
            The label of the y-axis.
        legend : string
            The legend of the plot.

        Returns
        -------
        plot : matplotlib plot
            The plot.
        """
        if len(components) == 2:
            ax = plt.figure().add_subplot()
            if title is not None:
                ax.set_title(title)
            if xlabel is not None:
                ax.set_xlabel(xlabel)
            if ylabel is not None:
                ax.set_ylabel(ylabel)
            if legend is not None:
                ax.legend(legend)
            plot = ax.plot(
                self.solution[:, components[0]], self.solution[:, components[1]], **kwargs)
        elif len(components) == 3:
            ax = plt.figure().add_subplot(projection='3d')
            if title is not None:
                ax.set_title(title)
            if xlabel is not None:
                ax.set_xlabel(xlabel)
            if ylabel is not None:
                ax.set_ylabel(ylabel)
            if zlabel is not None:
                ax.set_zlabel(zlabel)
            if legend is not None:
                ax.legend(legend)
            plot = ax.plot(self.solution[:, components[0]], self.solution[:,
                           components[1]], self.solution[:, components[2]], **kwargs)

        return plot
