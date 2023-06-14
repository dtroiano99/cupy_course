import numpy as np
from cupyx.profiler import benchmark
import nbody_const

class NumpySimulator():
    def __init__(self, pos, vel) -> None:
        self.positions = pos
        self.velocities = vel

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

        # in the last line, we scale the forces by multiplying them with the product 
        # of G (the gravitational constant). 
        self.forces *= nbody_const.G

    def update_positions(self) -> None:
        accelerations = self.forces

        # Update velocities based on accelerations and time step
        self.velocities = self.velocities + accelerations * nbody_const.DT

        # Update positions based on new velocities and time step
        self.positions = self.positions + self.velocities * nbody_const.DT


    def simulate_n_body(self) -> None:
        # Perform simulation for the specified number of steps
        for _ in range(nbody_const.N_STEPS):
            # Calculate forces acting on the bodies
            self.calculate_forces()

            # Update positions and velocities based on the forces and time step
            self.update_positions()

    def run_benchmark(self) -> None:
        bench = benchmark(self.simulate_n_body, (), n_repeat=10)
        print(bench.to_str())
