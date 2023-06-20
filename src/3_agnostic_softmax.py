import cupy as cp
import numpy as np
from cupyx.profiler import benchmark

# Stable implementation of log(1 + exp(x))
def softplus(x):
    xp = cp.get_array_module(x)
    print("Using:", xp.__name__)
    return xp.maximum(0, x) + xp.log1p(xp.exp(-abs(x)))

x = np.random.random(int(1e5))
x_gpu = cp.asarray(x)

cpu_bench = benchmark(softplus, (x,), n_repeat=10)
gpu_bench = benchmark(softplus, (x_gpu,), n_repeat=10)

print(f'CPU version timing: {cpu_bench.to_str()}')
print(f'GPU version timing: {gpu_bench}')