import numpy as np

import nbody_const
from nbody_numpy import NumpySimulator

positions = np.random.rand(nbody_const.N_BODIES, nbody_const.N_DIM)
velocities = np.random.rand(nbody_const.N_BODIES, nbody_const.N_DIM)

# instance the simulator object
np_simulator = NumpySimulator(positions, velocities)

# # Run the simulation
# np_simulator.simulate_n_body(positions, velocities, 
#                              nbody_const.G, nbody_const.DT, nbody_const.N_STEPS)

# Benchmark the simulation
np_simulator.run_benchmark()