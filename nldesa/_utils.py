import jax.numpy as jnp
from ._objects import EquationSystem, StabilityAnalysis

def StabilityMap(f, y0, t0, t1, n, params, component):
    r = []
    for i in range(0, len(params[0])):
        for j in range(0, len(params[1])):
            S = StabilityAnalysis(f, y0, t0, t1, n, jnp.asarray([params[0][i], params[1][j]]))
            S.DMD(component)
            status = S.stability(component)
            temp = jnp.asarray([params[0][i], params[1][j], status])
            r.append(temp) 
    r = jnp.asarray(r) 
    return r