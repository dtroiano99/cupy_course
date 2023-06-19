import cupy as cp
import numpy as np
import nbody_const
import nbody_agnostic as ag_nbody

positions = np.random.randn(nbody_const.N_BODIES, nbody_const.N_DIM)
velocities = np.random.randn(nbody_const.N_BODIES, nbody_const.N_DIM)
masses = np.random.randn(nbody_const.N_BODIES)

gpu_positions = cp.asarray(positions)
gpu_velocities = cp.asarray(velocities)
gpu_masses = cp.asarray(masses)

ag_positions_np = ag_nbody.simulate_n_body(positions, masses, velocities)
ag_positions_cp = ag_nbody.simulate_n_body(gpu_positions, gpu_masses, gpu_velocities)

cpu_gpu_pos = ag_positions_cp.get()

if np.allclose(ag_positions_np, cpu_gpu_pos):
    print('The positions computed on GPU are the same as the ones computed on CPU. Well done.')
else:
    print('The positions computed on GPU are NOT the same as the ones computed on CPU. Check the code.')