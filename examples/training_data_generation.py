import jax.numpy as jnp
from jax import jit
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import time as time

from nldesa import StabilityMap, StabilityMap_Plot

NUMBER_OF_CORES = 8

init_time = time.time()
 
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
N = 100
component = 0

if __name__ == '__main__': # Do not remove, to allow multiproessing in windows
    # Define the parameters

    num_sets = 10
    set_size = 10000
    for i in range(0,num_sets):
        
        random_min = np.random.uniform(low=-100, high=-95)
        random_max = np.random.uniform(low=95, high=100)
        
        a0 = jnp.linspace(random_min, random_max, int(np.sqrt(set_size)))
        a1 = jnp.linspace(random_min, random_max, int(np.sqrt(set_size)))
        a = jnp.asarray([a0, a1])
        # First column are the values of a[0], second column are the values of a[1]

        Map = StabilityMap(system, y0, T_0, T_1, N, a, component,NUMBER_OF_CORES)

        new_features = Map[:, :2]
        new_labels = Map[:, 2]
        
        try:
            # Load the existing data from the npz file
            existing_data = np.load('data/training_set.npz')
            existing_features = existing_data['features']
            existing_labels = existing_data['labels']

            # Concatenate the new data with the existing data
            combined_features = np.concatenate([existing_features, new_features])
            combined_labels = np.concatenate([existing_labels, new_labels])

            # Save the combined data to a new npz file
            np.savez('data/training_set.npz', features=combined_features, labels=combined_labels)
        except:
            np.savez('data/training_set.npz', features=new_features, labels=new_labels)
            
        
        print(f"\033[92m {set_size} new points have been succesfully generated! \033[0m")
        print(f"\033[92m Total number of points: {combined_labels.size} \033[0m")
        
        print(f"\033[94m Current set: {i+1}/{num_sets} \033[0m")
        
        final_time = time.time() 
        print(f"Total time: {final_time - init_time}")