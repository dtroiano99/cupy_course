from cupyx.profiler import benchmark
import cupy as cp
import numpy as np
 
# NumPy and CPU Runtime
def cpu_init():
    return np.ones((1000, 1000, 200))
 
# CuPy and GPU Runtime
def gpu_init():
    return cp.ones((1000, 1000, 200))

cpu_bench = benchmark(cpu_init, n_repeat=20)
gpu_bench = benchmark(gpu_init, n_repeat=20)

print(f'CPU version timing: {cpu_bench.to_str()}')
print(f'GPU version timing: {gpu_bench.to_str()}')