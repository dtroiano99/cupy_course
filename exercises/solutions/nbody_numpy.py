import numpy as np

N_BODIES = 1000
N_DIM = 3
G = 6.67430e-11  # gravitational constant
DT = 0.01  # time step
N_STEPS = 1000

class np_simulator():
    def __init__(self, pos, vel, masses) -> None:
        self.positions = pos
        self.velocities = vel
        self.masses = masses

    def calculate_forces(self) -> None:
        # positions[:, np.newaxis, :] is used to introduce a new axis to the positions array.
        # The resulting shape of positions_diff will be (N, 1, 3) where N is the number of bodies.
        # By subtracting positions from positions[:, np.newaxis, :], we obtain the difference 
        # in positions between each pair of bodies. The resulting positions_diff array 
        # will have shape (N, N, 3), where N is the number of bodies, 
        # and positions_diff[i, j] represents the difference in position between bodies i and j.
        positions_diff = self.positions[:, np.newaxis, :] - self.positions

        # np.linalg.norm() is used to calculate the Euclidean distance between 
        # each pair of bodies based on the positions_diff array. 
        # The axis=2 argument indicates that the norm is computed along 
        # the third axis of the array, resulting in a 2D array distances of shape (N, N).
        distances = np.linalg.norm(positions_diff, axis=2)

        # To handle division by zero, np.nan_to_num is used, arbitrarily setting nan to zeroes.
        inv_distances_cubed = np.nan_to_num(1.0 / distances ** 3, nan=0)

        # Here, inv_distances_cubed[:, :, np.newaxis] is used to introduce 
        # a new axis to the inv_distances_cubed array. The resulting shape 
        # will be (N, N, 1). This new axis allows us to perform element-wise 
        # multiplication with the positions_diff array, which has shape (N, N, 3).
        self.forces = np.sum(inv_distances_cubed[:, :, np.newaxis] * (positions_diff), axis=1)

        # n the last line, we scale the forces by multiplying them with the product 
        # of G (the gravitational constant) and the masses of the bodies. 
        # masses[:, np.newaxis] is used to introduce a new axis to the masses array, 
        # resulting in a shape (N, 1). This allows for element-wise multiplication with 
        # the forces array, which has shape (N, 3). 
        # The resulting forces will be scaled by the gravitational constant and the masses.
        self.forces *= G * self.masses[:, np.newaxis]

    def update_positions(self) -> None:
        # Calculate accelerations by dividing forces by masses
        accelerations = self.forces / self.masses[:, np.newaxis]

        # Update velocities based on accelerations and time step
        self.velocities = self.velocities + accelerations * DT

        # Update positions based on new velocities and time step
        self.positions = self.positions + self.velocities * DT


    def simulate_n_body(self) -> None:
        # Perform simulation for the specified number of steps
        for _ in range(N_STEPS):
            # Calculate forces acting on the bodies
            self.calculate_forces()

            # Update positions and velocities based on the forces and time step
            self.update_positions()


# positions = np.random.rand(N_BODIES, N_DIM)
# velocities = np.random.rand(N_BODIES, N_DIM)
# masses = np.random.rand(N_BODIES)

# # Run the simulation
# final_positions, final_velocities = simulate_n_body(positions, velocities, masses, G, DT, N_STEPS)