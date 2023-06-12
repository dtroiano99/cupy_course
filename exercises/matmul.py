import numpy as np
import cupy as cp

# matrixes on CPU
a = np.random.randint(0, 10, (1_000, 5_000))
b = np.random.randint(0, 10, (5_000, 1_000))

# TODO: code to use the same matrixes on GPU

# TODO: matrix multiplication between a and b with numpy and cupy

# TODO: compare results and verify they are all close

# TODO: Try to change array dimensions and profile the code using the benchmark tool from cupy