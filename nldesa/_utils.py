import jax.numpy as jnp
from ._objects import EquationSystem, StabilityAnalysis
import matplotlib.pyplot as plt

from multiprocessing.pool import Pool

""" Stability Map Computations """

def StabilityMap_Computations(f,y0,t0,t1,n,params,component):
    S = StabilityAnalysis(f, y0, t0, t1, n, jnp.asarray([params[0], params[1]]))
    S.DMD(component)
    status = S.stability(component)
    return jnp.asarray([params[0], params[1], status])

def StabilityMap(f,y0,t0,t1,n,params,component,num_cores = None):
    """ If num_cores (number of cores) is left blank, the maximum number of cores available are used"""
    Params_0, Params_1 = jnp.meshgrid(params[0],params[1])
    params_mesh = jnp.asarray(jnp.column_stack((Params_0.ravel(), Params_1.ravel())))

    r = []
    with Pool(num_cores) as pool:
        args = [(f, y0, t0, t1, n, coordinate, component) for coordinate in params_mesh]
        for result in pool.starmap(StabilityMap_Computations, args):
            r.append(result)
    
    r = jnp.asarray(r) 
    print(r)
    return r

def StabilityMap_Plot(data):
    x= jnp.unique(data[:,0])
    y= jnp.unique(data[:,1])
    X,Y = jnp.meshgrid(x,y)
    
    z = data[:,2]
    Z=z.reshape(len(y),len(x))

    plt.pcolormesh(X,Y,Z)

    plt.show()