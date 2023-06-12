import cupy as cp

N_BODIES = 1000
N_DIM = 3
G = 6.67430e-11  # gravitational constant
DT = 0.01  # time step
N_STEPS = 1000

class agnostic_simulator():
    def __init__(self, pos, vel, masses) -> None:
        self.positions = pos
        self.velocities = vel
        self.masses = masses

    def simulate_nbody(self):
        xp = cp.get_array_module(self.positions)
        print(f'using: {xp.__name__}')

        def _calculate_forces(self) -> None:
            positions_diff = self.positions[:, xp.newaxis, :] - self.positions

            distances = xp.linalg.norm(positions_diff, axis=2)

            inv_distances_cubed = xp.nan_to_num(1.0 / distances ** 3, nan=0)

            self.forces = xp.sum(inv_distances_cubed[:, :, xp.newaxis] * (positions_diff), axis=1)

            self.forces *= G * self.masses[:, xp.newaxis]

        def _update_positions(self) -> None:
            accelerations = self.forces / self.masses[:, xp.newaxis]

            self.velocities = self.velocities + accelerations * DT
            self.positions = self.positions + self.velocities * DT

        for _ in range(N_STEPS):
            _calculate_forces()
            _update_positions()

# positions = cp.random.rand(N_BODIES, N_DIM)
# velocities = cp.random.rand(N_BODIES, N_DIM)
# masses = cp.random.rand(N_BODIES)

# pos, vel = simulate_nbody(positions, velocities, masses)