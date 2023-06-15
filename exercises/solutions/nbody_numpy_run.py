import numpy as np
import nbody_const
import nbody_numpy as np_sim
from cupyx.profiler import benchmark

positions = np.random.randn(nbody_const.N_BODIES, nbody_const.N_DIM)
velocities = np.random.randn(nbody_const.N_BODIES, nbody_const.N_DIM)
masses = 20.0 * np.ones((nbody_const.N_BODIES, 1)) / nbody_const.N_BODIES  # total mass of particles is 20

# Run the simulation
# np_positions = np_sim.simulate_n_body(positions, masses, velocities)

# Benchmark the simulation
bench = benchmark(np_sim.simulate_n_body, (positions, masses, velocities), n_repeat=10)
print(bench.to_str())
