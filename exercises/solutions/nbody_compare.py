import numpy as np
import cupy as cp

from cupyx.profiler import benchmark

import nbody_numpy as np_sim
import nbody_cupy as cp_sim
import nbody_agnostic as ag_sim

import nbody_const

positions = np.random.randn(nbody_const.N_BODIES, nbody_const.N_DIM)
velocities = np.random.randn(nbody_const.N_BODIES, nbody_const.N_DIM)
masses = 20.0 * np.ones((nbody_const.N_BODIES, 1)) / nbody_const.N_BODIES  # total mass of particles is 20

gpu_positions = cp.asarray(positions)
gpu_velocities = cp.asarray(velocities)
gpu_masses = cp.asarray(masses)

np_positions = np_sim.simulate_n_body(positions, masses, velocities)
cp_positions = cp_sim.simulate_n_body(gpu_positions, gpu_masses, gpu_velocities)
cpu_gpu_pos = cp_positions.get()
# print(cpu_gpu_pos[:-10])
print( np.allclose(np_positions, cpu_gpu_pos))
print(np_positions - cpu_gpu_pos)

# np2_positions = np_sim.simulate_n_body(positions, masses, velocities)
# print( np.allclose(np2_positions, np_positions))
# ag_positions = ag_sim.simulate_n_body(gpu_positions, gpu_masses, gpu_velocities)

cp2_positions = cp_sim.simulate_n_body(gpu_positions, gpu_masses, gpu_velocities)
cpu_gpu_pos2 = cp2_positions.get()
print( np.allclose(np_positions, cpu_gpu_pos2))
print(np_positions - cpu_gpu_pos2)
#cpu_gpu_pos = cp_positions.get()

# print(f"First ten elements of numpy positions: {np_positions[:10]}")
# print(f"First ten elements of cupy positions: {cp_positions[:10]}")

# print(cpu_gpu_pos[:-10])
print( np.allclose(np_positions, cpu_gpu_pos))
# print( np.allclose(np_positions, cpu_gpu_pos2))
# print( np.allclose(cpu_gpu_pos2, cpu_gpu_pos))
# print( np.allclose(np2_positions, cpu_gpu_pos))
# print( np.allclose(np2_positions, cpu_gpu_pos2))