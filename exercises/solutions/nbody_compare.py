import numpy as np
import cupy as cp

from cupyx.profiler import benchmark

from nbody_numpy import NumpySimulator
from nbody_cupy import CupySimulator
from nbody_agnostic import AgnosticSimulator

import nbody_const

positions = np.random.rand(nbody_const.N_BODIES, nbody_const.N_DIM)
velocities = np.random.rand(nbody_const.N_BODIES, nbody_const.N_DIM)

gpu_positions = cp.asarray(positions)
gpu_velocities = cp.asarray(velocities)

numpy_sim = NumpySimulator(positions, velocities)
cupy_sim = CupySimulator(gpu_positions, gpu_velocities)
ag_sim = AgnosticSimulator(gpu_positions, gpu_velocities)

numpy_sim.simulate_n_body()
cupy_sim.simulate_n_body()
ag_sim.simulate_n_body()

cpu_gpu_pos = cupy_sim.positions.get()
cpu_gpu_vel = cupy_sim.velocities.get()

print(f"First ten elements of numpy positions: {numpy_sim.positions[:10]}")
print(f"First ten elements of cupy positions: {cpu_gpu_pos[:10]}")

assert np.allclose(numpy_sim.positions, cpu_gpu_pos)
assert np.allclose(numpy_sim.velocities, cpu_gpu_vel)