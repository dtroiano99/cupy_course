import cupy as cp
import numpy as np

# Generate sample data
np.random.seed(0)
x = np.linspace(0, 10, 100)
y_true = 2 * np.sin(x) + 0.5 * x
noise = np.random.normal(0, 0.5, x.shape)
y = y_true + noise

# TODO: put the code developed for polyfit.py into a function,
# code the same function for numpy, and compare the execution time
# using the cupyx.profiler.benchmark function. Compare also the prediction
# and check they are all close.