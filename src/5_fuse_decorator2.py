import cupy as cp
from cupyx.profiler import benchmark

@cp.fuse(kernel_name='squared_diff')
def squared_diff(x, y):
    return (x - y) * (x - y)

b = cp.arange(10, 20)
c = cp.arange(20, 30)

bench = benchmark(squared_diff, (b, c), n_repeat=10)

print(bench)