import cupy as cp
import numpy as np

# Generate sample data
np.random.seed(0)
x = np.linspace(0, 10, 100)
y_true = 2 * np.sin(x) + 0.5 * x
noise = np.random.normal(0, 0.5, x.shape)
y = y_true + noise

# TODO: Convert data to CuPy arrays

# TODO: Perform polynomial fitting using cp.polyfit()
# You can see function signatures in ipython by typing FUNC_NAME? and then pressing enter
# degree = 5
# coeffs =

# TODO: Generate predictions using the fitted polynomial
# y_pred =

# TODO: Compute the sum of squared residuals
# ssr = 

# TODO: Compute the total sum of squares
# tss = 

# TODO: Compute the coefficient of determination (R-squared)
# r_squared = 


# print(f"R-squared: {r_squared}")