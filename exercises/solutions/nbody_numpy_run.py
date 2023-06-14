import numpy as np

import nbody_const
from nbody_numpy import NumpySimulator

positions = np.random.randn(nbody_const.N_BODIES, nbody_const.N_DIM)
velocities = np.random.randn(nbody_const.N_BODIES, nbody_const.N_DIM)
masses = 20.0 * np.ones((nbody_const.N_BODIES, 1)) / nbody_const.N_BODIES  # total mass of particles is 20

# instance the simulator object
np_simulator = NumpySimulator(positions, masses)

# # Run the simulation
# np_simulator.simulate_n_body(positions, velocities, 
#                              nbody_const.G, nbody_const.DT, nbody_const.N_STEPS)

# Benchmark the simulation
np_simulator.run_benchmark(velocities)