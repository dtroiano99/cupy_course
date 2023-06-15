import cupy as cp
import nbody_const
import nbody_cupy as cp_sim
from cupyx.profiler import benchmark

positions = cp.random.randn(nbody_const.N_BODIES, nbody_const.N_DIM)
velocities = cp.random.randn(nbody_const.N_BODIES, nbody_const.N_DIM)
masses = 20.0 * cp.ones((nbody_const.N_BODIES, 1)) / nbody_const.N_BODIES  # total mass of particles is 20

# Run the simulation
# cp_positions = cp_sim.simulate_n_body(positions, masses, velocities)

# Benchmark the simulation
bench = benchmark(cp_sim.simulate_n_body, (positions, masses, velocities), n_repeat=10)
print(bench.to_str())