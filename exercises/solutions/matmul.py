import numpy as np
import cupy as cp

# matrixes on CPU
a = np.random.randint(0, 10, (1_000, 5_000))
b = np.random.randint(0, 10, (5_000, 1_000))

# write here code to use the same matrixes on GPU
gpu_a = cp.asarray(a)
gpu_b = cp.asarray(b)

# write code for matrix multiplication between a and b with numpy and cupy
c = np.matmul(a, b)
gpu_c = cp.matmul(a, b)

# compare results and verify they are all close
cpu_gpu_c = gpu_c.get()

assert np.allclose(c, cpu_gpu_c)