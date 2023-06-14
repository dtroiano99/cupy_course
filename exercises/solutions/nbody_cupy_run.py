import cupy as cp

import nbody_const
from nbody_cupy import CupySimulator

positions = cp.random.randn(nbody_const.N_BODIES, nbody_const.N_DIM)
velocities = cp.random.randn(nbody_const.N_BODIES, nbody_const.N_DIM)
masses = 20.0 * cp.ones((nbody_const.N_BODIES, 1)) / nbody_const.N_BODIES  # total mass of particles is 20

# instance the simulator object
cp_simulator = CupySimulator(positions, masses)

# # Run the simulation
# np_simulator.simulate_n_body(positions, velocities, 
#                              nbody_const.G, nbody_const.DT, nbody_const.N_STEPS)

# Benchmark the simulation
cp_simulator.run_benchmark(velocities)