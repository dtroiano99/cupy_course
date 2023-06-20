import cupy as cp
from cupyx.profiler import benchmark

# Define input arrays
a = cp.arange(10)
b = cp.arange(10, 20)
c = cp.arange(20, 30)

# Define an elementwise computation using @cp.fuse() decorator
@cp.fuse()
def elementwise_computation(x, y, z):
    return cp.sin(x) + cp.cos(y) / cp.sqrt(z)

# Invoke the elementwise computation
bench = benchmark(elementwise_computation, (a, b, c), n_repeat=10)

# Print the result
print(bench)