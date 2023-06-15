import cupy as cp
import nbody_const
import nbody_cupy as cp_sim
from cupyx.profiler import benchmark

# TODO: instantiate arrays for positions, velocities, masses using
# nbody_const constants. The pos and vel arrays must have shape (N_BODIES, N_DIM),
# masses must have shape (N_BODIES)
positions = 
velocities = 
masses = 

# Benchmark the simulation
# you can use the np_sim namespace to call the proper function
# https://docs.cupy.dev/en/stable/user_guide/performance.html#benchmarking
bench = 

print(bench.to_str())
