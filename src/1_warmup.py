import cupy as cp
import numpy as np

import time
 
# NumPy and CPU Runtime
cpus = time.perf_counter()
np.ones((1000, 1000, 200))
cpue = time.perf_counter()
print(f"Time consumed by numpy: {cpue - cpus}")
 
# CuPy and GPU Runtime
s = time.perf_counter()
cp.ones((1000, 1000, 200))
e = time.perf_counter()
print(f"\nTime consumed by cupy: {e - s}")

print(f"\nspeed-up is by a factor {(cpue-cpus)/(e-s)}")