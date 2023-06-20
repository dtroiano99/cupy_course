import numpy as np
import cupy as cp
from cupyx.profiler import benchmark
# matrixes on CPU
a = np.random.randint(0, 10, (1_000, 5_000))
b = np.random.randint(0, 10, (5_000, 1_000))

# write here code to use the same matrixes on GPU
gpu_a = cp.asarray(a)
gpu_b = cp.asarray(b)

# write code for matrix multiplication between a and b with numpy and cupy
c = np.matmul(a, b)
gpu_c = cp.matmul(gpu_a, gpu_b)

# compare results and verify they are all close
cpu_gpu_c = gpu_c.get()

if np.allclose(c, cpu_gpu_c):
    print('The two arrays are all close, well done!')
else:
    print('We have a problem, the two arrays are not all close, check your code')

cpu_bench = benchmark(np.matmul, (a, b), n_repeat=10)
gpu_bench = benchmark(cp.matmul, (gpu_a, gpu_b), n_repeat=10)

print(f'CPU time: {cpu_bench.to_str()})')
print(f'GPU time: {gpu_bench})')