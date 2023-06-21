import numpy as np
import nbody_const
# from cupyx.profiler import benchmark

def getAcc(positions, mass): 
    # Extract x, y, z components of positions
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]
    
    # Calculate pairwise separations
    # np.newaxis is a special constant in NumPy that is used to 
    # increase the dimensions of an array by one. It is often used 
    # to convert a 1D array into a 2D array.
    # The result of this operation is a 2D array where each element 
    # represents the difference between the corresponding elements in x. 
    # The shape of the resulting array will be (n, n), where n is the 
    # number of elements in x. The element at position (i, j) in the 
    # resulting array will be x[i] - x[j].
    dx = x[:, np.newaxis] - x
    dy = y[:, np.newaxis] - y
    dz = z[:, np.newaxis] - z
    
    # matrix that stores 1/r^3 for all particle pairwise particle separations 
    inv_r3 = np.sqrt(dx**2 + dy**2 + dz**2 + 0.01**2)
    inv_r3[inv_r3>0] = inv_r3[inv_r3>0]**(-3)

    ax = nbody_const.G * np.matmul((dx * inv_r3), mass)
    ay = nbody_const.G * np.matmul((dy * inv_r3), mass)
    az = nbody_const.G * np.matmul((dz * inv_r3), mass)
    
    # pack together the acceleration components
    a = np.column_stack((ax, ay, az))

    return a

def simulate_n_body(positions, mass, v0):
    # Convert to Center-of-Mass frame
    vel = v0 - np.mean(mass[:, np.newaxis] * v0, 0) / np.mean(mass)
    # calculate initial gravitational accelerations
    acc = getAcc(positions, mass)
    
    for _ in range(nbody_const.N_STEPS):
        # (1/2) kick
        vel += acc * nbody_const.DT/2.0
        # drift
        positions += vel * nbody_const.DT
        # update accelerations
        acc = getAcc(positions, mass)
        # (1/2) kick
        vel += acc * nbody_const.DT/2.0
    
    return positions

# bench = benchmark(simulate_n_body, (positions, mass, velocities), n_repeat=10)
# print(bench.to_str())
