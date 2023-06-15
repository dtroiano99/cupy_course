import cupy as cp
import nbody_const
import nbody_cupy as cp_sim
from cupyx.profiler import benchmark

positions = cp.random.randn(nbody_const.N_BODIES, nbody_const.N_DIM)
velocities = cp.random.randn(nbody_const.N_BODIES, nbody_const.N_DIM)
masses = cp.random.randn(nbody_const.N_BODIES)

# Benchmark the simulation
bench = benchmark(cp_sim.simulate_n_body, (positions, masses, velocities), n_repeat=10)
print(bench.to_str())