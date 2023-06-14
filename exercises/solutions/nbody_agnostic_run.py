import cupy as cp

import nbody_const
from nbody_agnostic import AgnosticSimulator

positions = cp.random.rand(nbody_const.N_BODIES, nbody_const.N_DIM)
velocities = cp.random.rand(nbody_const.N_BODIES, nbody_const.N_DIM)
masses = 20.0 * cp.ones((nbody_const.N_BODIES, 1)) / nbody_const.N_BODIES  # total mass of particles is 20

# instance the simulator object
ag_simulator = AgnosticSimulator(positions, velocities)

# Run the simulation
ag_simulator.simulate_n_body()