import cupy as cp
from cupyx import jit

@jit.rawkernel()
def elementwise_square(x, y, size):
    tid = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    ntid = jit.gridDim.x * jit.blockDim.x
    for i in range(tid, size, ntid):
        y[i] = x[i] * x[i]

size = cp.uint32(2 ** 22)
x = cp.arange(size, dtype=cp.float32)
y = cp.empty((size,), dtype=cp.float32)

elementwise_square((128,), (1024,), (x, y, size))  # RawKernel style
assert (y == x * x).all()

elementwise_square[128, 1024](x, y, size)  # Numba style
assert (y == x * x).all()