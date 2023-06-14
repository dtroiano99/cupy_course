import cupy as cp

import nbody_const
from nbody_agnostic import AgnosticSimulator

positions = cp.random.rand(nbody_const.N_BODIES, nbody_const.N_DIM)
velocities = cp.random.rand(nbody_const.N_BODIES, nbody_const.N_DIM)

# instance the simulator object
ag_simulator = AgnosticSimulator(positions, velocities)

# Run the simulation
ag_simulator.simulate_n_body()