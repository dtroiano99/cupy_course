import cupy as cp

N_BODIES = 1000
N_DIM = 3
G = 6.67430e-11  # gravitational constant
DT = 0.01  # time step
N_STEPS = 1000

class cp_simulator():
    def __init__(self, pos, vel, masses) -> None:
        self.positions = pos
        self.velocities = vel
        self.masses = masses

    def calculate_forces(self) -> None:
        positions_diff = self.positions[:, cp.newaxis, :] - self.positions

        distances = cp.linalg.norm(positions_diff, axis=2)

        inv_distances_cubed = cp.nan_to_num(1.0 / distances ** 3, nan=0)

        self.forces = cp.sum(inv_distances_cubed[:, :, cp.newaxis] * (positions_diff), axis=1)

        self.forces *= G * self.masses[:, cp.newaxis]

    def update_positions(self) -> None:
        accelerations = self.forces / self.masses[:, cp.newaxis]

        self.velocities = self.velocities + accelerations * DT
        self.positions = self.positions + self.velocities * DT

    def simulate_n_body(self) -> None:
        for _ in range(N_STEPS):
            self.calculate_forces()
            self.update_positions()


# positions = cp.random.rand(N_BODIES, N_DIM)
# velocities = cp.random.rand(N_BODIES, N_DIM)
# masses = cp.random.rand(N_BODIES)

# # Run the simulation
# final_positions, final_velocities = simulate_n_body(positions, velocities, masses, G, DT, N_STEPS)