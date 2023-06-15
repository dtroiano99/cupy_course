import numpy as np
import cupy as cp
from cupyx.profiler import benchmark

# matrixes on CPU
a = np.random.randint(0, 10, (1_000, 5_000))
b = np.random.randint(0, 10, (5_000, 1_000))

# TODO: code to use the same matrixes on GPU
gpu_a = 
gpu_b = 

# TODO: matrix multiplication between a and b with numpy and cupy
c = 
gpu_c = 

# TODO: take the gpu array to CPU
cpu_gpu_c =

# TODO: compare results and verify they are all close


# TODO: Try to change array dimensions and profile the code using the benchmark tool from cupy