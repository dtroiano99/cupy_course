import numpy as np
import cupy as cp

from cupyx.profiler import benchmark

import nbody_numpy as np_sim
import nbody_cupy as cp_sim

import nbody_const

positions = np.random.randn(nbody_const.N_BODIES, nbody_const.N_DIM)
velocities = np.random.randn(nbody_const.N_BODIES, nbody_const.N_DIM)
masses = np.random.randn(nbody_const.N_BODIES)

gpu_positions = cp.asarray(positions)
gpu_velocities = cp.asarray(velocities)
gpu_masses = cp.asarray(masses)

np_positions = np_sim.simulate_n_body(positions, masses, velocities)
cp_positions = cp_sim.simulate_n_body(gpu_positions, gpu_masses, gpu_velocities)
cpu_gpu_pos = cp_positions.get()

if np.allclose(np_positions, cpu_gpu_pos):
    print('The positions computed on GPU are the same as the ones computed on CPU. Well done.')
else:
    print('The positions computed on GPU are NOT the same as the ones computed on CPU. Check the code.')