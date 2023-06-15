import cupy as cp
import numpy as np
import nbody_const
import nbody_agnostic as ag_nbody

# TODO: instantiate arrays for positions, velocities, masses using
# nbody_const constants. The pos and vel arrays must have shape (N_BODIES, N_DIM),
# masses must have shape (N_BODIES)
positions = 
velocities = 
masses = 

# TODO: get the previous array instantiated on CPU to the device
gpu_positions = 
gpu_velocities = 
gpu_masses = 

# TODO: run the simulator with CPU inputs and with GPU inputs
ag_positions_np = 
ag_positions_cp = 

# TODO: get the results from GPU run to CPU
cpu_gpu_pos = 

# compare the results
if np.allclose(ag_positions_np, cpu_gpu_pos):
    print('The positions computed on GPU are the same as the ones computed on CPU. Well done.')
else:
    print('The positions computed on GPU are NOT the same as the ones computed on CPU. Check the code.')
