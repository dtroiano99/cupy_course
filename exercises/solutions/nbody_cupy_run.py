import cupy as cp

import nbody_const
from nbody_cupy import CupySimulator

positions = cp.random.rand(nbody_const.N_BODIES, nbody_const.N_DIM)
velocities = cp.random.rand(nbody_const.N_BODIES, nbody_const.N_DIM)

# instance the simulator object
cp_simulator = CupySimulator(positions, velocities)

# # Run the simulation
# np_simulator.simulate_n_body(positions, velocities, 
#                              nbody_const.G, nbody_const.DT, nbody_const.N_STEPS)

# Benchmark the simulation
cp_simulator.run_benchmark()