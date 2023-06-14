import cupy as cp
import nbody_const

from cupyx.profiler import benchmark

class AgnosticSimulator():
    def __init__(self, pos, vel) -> None:
        self.positions = pos
        self.velocities = vel
        self.xp = cp.get_array_module(self.positions)
        print(f'using: {self.xp.__name__}')

    def calculate_forces(self) -> None:
        positions_diff = self.positions[:, self.xp.newaxis, :] - self.positions
        distances = self.xp.linalg.norm(positions_diff, axis=2)
        inv_distances_cubed = self.xp.nan_to_num(1.0 / distances ** 3, nan=0)
        self.forces = self.xp.sum(inv_distances_cubed[:, :, self.xp.newaxis] * (positions_diff), axis=1)
        
    def update_positions(self) -> None:
        accelerations = self.forces
        self.velocities = self.velocities + accelerations * nbody_const.DT
        self.positions = self.positions + self.velocities * nbody_const.DT

    def simulate_n_body(self):
        for _ in range(nbody_const.N_STEPS):
            self.calculate_forces()
            self.update_positions()

    # def run_benchmark(self) -> None:
    #     bench = benchmark(self.simulate_n_body, (), n_repeat=10)
    #     print(bench.to_str())
