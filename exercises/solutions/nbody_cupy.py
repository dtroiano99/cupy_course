import cupy as cp
import nbody_const
from cupyx.profiler import benchmark

N_BODIES = 1000
N_DIM = 3
G = 6.67430e-11  # gravitational constant
DT = 0.01  # time step
N_STEPS = 1000

class CupySimulator():
    def __init__(self, pos, vel) -> None:
        self.positions = pos
        self.velocities = vel

    def calculate_forces(self) -> None:
        positions_diff = self.positions[:, cp.newaxis, :] - self.positions

        distances = cp.linalg.norm(positions_diff, axis=2)

        inv_distances_cubed = cp.nan_to_num(1.0 / distances ** 3, nan=0)

        self.forces = cp.sum(inv_distances_cubed[:, :, cp.newaxis] * (positions_diff), axis=1)

    def update_positions(self) -> None:
        accelerations = self.forces

        self.velocities = self.velocities + accelerations * nbody_const.DT
        self.positions = self.positions + self.velocities * nbody_const.DT

    def simulate_n_body(self) -> None:
        for _ in range(nbody_const.N_STEPS):
            self.calculate_forces()
            self.update_positions()

    def run_benchmark(self) -> None:
        bench = benchmark(self.simulate_n_body, (), n_repeat=10)
        print(bench.to_str())
