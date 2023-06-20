import cupy as cp
import nbody_const

def getAcc(positions, mass): 
    """
    takes an array of particle positions (positions) and their masses (mass) as input. 
    It calculates the pairwise separations between particles, 
    computes the inverse cube distances, and then calculates 
    the acceleration components for each particle. 
    The resulting acceleration array is returned as the output of the function.
    """

    # TODO: Extract x, y, z components of positions.
    x = 
    y = 
    z = 
    
    # TODO: Calculate pairwise separations
    # Use broadcasting to subtract each pair of positions along the x, y, and z axes. 
    # The resulting arrays dx, dy, and dz store 
    # the separations between particles in each dimension.
    dx = 
    dy = 
    dz = 
    
    # TODO: matrix that stores 1/r^3 for all particle pairwise particle separations 
    # The inv_r3 matrix stores the inverse cube distances between particles. 
    # It first calculates the Euclidean distance between each pair of particles 
    # using the separations dx, dy, and dz. 
    # The distances are computed by taking the square root of the sum of squared separations. 
    # A small epsilon (0.01**2) is added to avoid division by zero. 
    # The resulting distances are then raised to the power of -3 
    # to obtain the inverse cube distances. 
    # Importantly, the elements of inv_r3 that are greater than zero 
    # are filtered using inv_r3>0 and then exponentiated to -3. 
    # This is to avoid division by zero and set the inverse cube distance 
    # to zero for self-interactions.
    inv_r3 = 

    # TODO: calculate the acceleration components using matrix multiplication. 
    # The ax, ay, and az arrays are obtained by performing matrix multiplication 
    # between dx, dy, and dz (multiplied by inv_r3) and the mass array. 
    # The multiplication is scaled by the gravitational constant nbody_const.G.
    ax = 
    ay = 
    az = 
    
    # TODO: The acceleration components (ax, ay, and az) are combined into 
    # a single array. Each row of the new array must represent 
    # the acceleration vector for a particle.
    a = 

    return a

def simulate_n_body(positions, mass, v0):
    """
    performs a simulation of an n-body system over a specified number of time steps. 
    It converts the initial velocities to the center-of-mass frame, 
    calculates the initial gravitational accelerations, and then updates 
    the positions and velocities using the leapfrog integration scheme within a loop. 
    The updated positions are returned as the final result of the simulation.
    """

    # TODO: Convert to Center-of-Mass frame
    # Subtract the mean velocity of the system weighted by the masses 
    # from the initial velocities (v0).
    vel = 
    
    # TODO: calculate initial gravitational accelerations
    # Use the getAcc function!
    acc = 
    
    # TODO: loop that runs for nbody_const.N_STEPS iterations. 
    # This loop represents the time steps of the simulation.

    for _ in range(nbody_const.N_STEPS):
        # TODO: (1/2) kick
        # velocities are updated by adding the gravitational accelerations
        # multiplied by half the time step (nbody_const.DT/2.0). 
        # This step corresponds to half of the velocity update in the leapfrog integration scheme.
        vel += 

        # TODO: drift
        # positions are updated by adding the velocities multiplied by the time step 
        # (nbody_const.DT). 
        # This step corresponds to the position update in the leapfrog integration scheme.
        positions += 
        
        # TODO: update accelerations
        # The gravitational accelerations are updated by calling 
        # the getAcc function again, using the updated 
        # positions and masses.
        acc = 
        
        # TODO: (1/2) kick
        # velocitiesare updated once again by adding the new gravitational accelerations 
        # multiplied by half the time step (nbody_const.DT/2.0). 
        # This step completes the velocity update for the current time step.
        vel += 
    
    return positions