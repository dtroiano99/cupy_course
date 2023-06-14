import cupy as cp
import nbody_const

from cupyx.profiler import benchmark

def getAcc(positions, mass):
    xp = cp.get_array_module(positions)

    # positions r = [x,y,z] for all particles
    x = positions[:, 0:1]
    y = positions[:, 1:2]
    z = positions[:, 2:3]
    
    # matrix that stores all pairwise particle separations: r_j - r_i
    dx = x.T - x
    dy = y.T - y
    dz = z.T - z  
    
    # matrix that stores 1/r^3 for all particle pairwise particle separations 
    inv_r3 = (dx**2 + dy**2 + dz**2 + 0.01**2)
    inv_r3[inv_r3>0] = inv_r3[inv_r3>0]**(-1.5)
    ax = nbody_const.G * xp.matmul((dx * inv_r3), mass)
    ay = nbody_const.G * xp.matmul((dy * inv_r3), mass)
    az = nbody_const.G * xp.matmul((dz * inv_r3), mass)
    
    # pack together the acceleration components
    a = xp.hstack((ax, ay, az))
    return a

def simulate_n_body(positions, mass, v0):
    xp = cp.get_array_module(positions)
    # Convert to Center-of-Mass frame
    vel = v0 - xp.mean(mass * v0, 0) / xp.mean(mass)
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
