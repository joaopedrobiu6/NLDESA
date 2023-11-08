from functools import partial
import jax.numpy as jnp
from jax.experimental.ode import odeint as jax_odeint
from jax import jit
from pydmd import DMD, HODMD
import numpy as np
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
        """ Plot the solution as a function of time.

        Parameters
        ----------
        component : int
            The component of the state vector to plot. For a 1 dimensional system of second order,
            0 is the function value, 1 is the 1st derivative, etc. For a 3 dimensional system of second order,
            {0, 1, 2} = {x, y, z}, {3, 4, 5} = {x', y', z'}, etc.
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

    def plot_by_component(self, components, title=None, xlabel=None, ylabel=None, zlabel=None, legend=None, **kwargs):
        """ Make a plot with the desired components [X = {x, y, z}, X' = {x', y', z'}, ...].
            - Phase portrait: components = [X, X'] or [X, X', X'', ...]
            - 3D plot: components = [0, 1, 2]
            - Trajetory: components = [X = {x, y, z}
            - Any combination desired of the outputs from the differential equation system.

        Parameters
        ----------
        components : array
            The components of the state vector to plot. For a 1 dimensional system of second order,
            0 is the function value, 1 is the 1st derivative, etc. For a 3 dimensional system of second order,
            {0, 1, 2} = {x, y, z}, {3, 4, 5} = {x', y', z'}, etc.
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
        if len(components) == 1 or components is int:
            print('Use plot_solution instead to plot a single component varying in time.')
            quit()
        elif len(components) == 2:
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
        else:
            print('Invalid number of components.')
            quit()

        return plot

class StabilityAnalysis(EquationSystem):
    pass
    """ Stability Analysis of a Differential Equation System 

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
        The number of time steps.~
    a : array
        The initial parameter vector.
    rtol : float
        The relative tolerance.
    atol : float
        The absolute tolerance.
    """

    def __init__(self, f, y0, t0, t1, n, a, rtol=1e-10, atol=1e-10):
        super().__init__(f, y0, t0, t1, n)
        """Computes the solution."""
        f_jit = jit(self.f)
        self.solution = jax_odeint(partial(f_jit, a=a), self.y0, self.t, rtol=rtol, atol=atol)
    
    def DMD(self, component):
        self.dmd = DMD(svd_rank=0, exact=True, opt=True).fit(self.solution[:, component][None])

    def HODMD(self, component):
        self.dmd = HODMD(svd_rank=1, exact=True, opt=True, d=1, svd_rank_extra=0).fit(self.solution[:, component][None])

    def solution(self):
        """ Return the solution of the equation system."""
        return self.solution
    
    def eigenvalues(self, component, absolute=True):
        """ Return the eigenvalues the Dynamic Mode Decomposition of the solution for a component.
        
        Parameters
        ----------
        component : int
            The component of the state vector to plot. For a 1 dimensional system of second order,
            0 is the function value, 1 is the 1st derivative, etc. For a 3 dimensional system of second order,
            {0, 1, 2} = {x, y, z}, {3, 4, 5} = {x', y', z'}, etc.
        
        Returns
        -------
        eigenvalues : array
            The eigenvalues. The first column is the real part and the second column is the imaginary part.
        """
        self.eigenv = jnp.vstack((jnp.asarray([self.dmd.eigs.real]), jnp.asarray([self.dmd.eigs.imag]))).T
        if absolute:
            return jnp.sqrt(self.eigenv[:, 0]**2 + self.eigenv[:, 1]**2)
        else:
            return self.eigenv

    def modes(self, component):
        hodmd = HODMD(svd_rank=0, exact=True, opt=True, d=30).fit(self.solution[:, component][None])
        self.modesv = hodmd.modes
        return self.modesv
    
    def amplitudes(self, component):
        hodmd = HODMD(svd_rank=0, exact=True, opt=True, d=30).fit(self.solution[:, component][None])
        self.amplitudesv = hodmd.dynamics
        return self.amplitudesv

    def plot_eigenvalues(self, component, title=None, xlabel=None, ylabel=None, legend=None, **kwargs1):
        """ Plot the eigenvalues of the Dynamic Mode Decomposition of the solution for a component.
        
        Parameters
        ----------
        component : int
            The component of the state vector to plot. For a 1 dimensional system of second order,
            0 is the function value, 1 is the 1st derivative, etc. For a 3 dimensional system of second order,
            {0, 1, 2} = {x, y, z}, {3, 4, 5} = {x', y', z'}, etc.
        title : string
            The title of the plot.
        xlabel : string
            The label of the x-axis.
        ylabel : string
            The label of the y-axis.
        legend : string
            The legend of the plot.
        **kwargs1 : dict
            Keyword arguments for the scatter plot.
        
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
        ax.scatter(self.eigenv[:, 0], self.eigenv[:, 1], **kwargs1)
        phi = jnp.linspace(0, 2*jnp.pi, 100)
        ax.plot(jnp.cos(phi), jnp.sin(phi), 'k--', alpha=0.5)
        ax.set_aspect('equal')
        ax.grid()
        return ax
    
    def stability(self, component):
        """ Return the stability of the solution for a component.

        Parameters
        ----------
        component : int
            The component of the state vector to plot. For a 1 dimensional system of second order,
            0 is the function value, 1 is the 1st derivative, etc. For a 3 dimensional system of second order,
            {0, 1, 2} = {x, y, z}, {3, 4, 5} = {x', y', z'}, etc.
        
        Returns
        -------
        stability : array
            The stability. <=0 is stable or deacaying, >0 is unstable.
        """

        self.eigenv = self.eigenvalues(component, absolute=False)
        mean_abs_value = jnp.mean(jnp.sqrt(self.eigenv[:, 0]**2 + self.eigenv[:, 1]**2))
        self.stability = jnp.where(jnp.abs(1-mean_abs_value) < 10e-5, 1, 0)
        return self.stability