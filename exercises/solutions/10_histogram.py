import math
import numpy as np
import cupy as cp
from cupyx.profiler import benchmark

# def histogram(input_array, output_array):
#     for item in input_array:
#         output_array[item] = output_array[item] + 1

def np_histogram(input_array, bins):
    hist, _ = np.histogram(input_array, bins=bins)
    return hist

def cp_histogram(input_array, bins):
    hist, _ = cp.histogram(input_array, bins=bins)
    return hist

# input size
size = 2**25

# allocate memory on CPU and GPU
input_gpu = cp.random.randint(256, size=size, dtype=cp.int32)
input_cpu = cp.asnumpy(input_gpu)
output_gpu = cp.zeros(256, dtype=cp.int32)
output_cpu = cp.asnumpy(output_gpu)

# CUDA code
histogram_cuda_code = r'''
extern "C"
__global__ void histogram(const int * input, int * output)
{
    int item = (blockIdx.x * blockDim.x) + threadIdx.x;
    __shared__ int temp_histogram[256];
 
    // Initialize shared memory and synchronize
    temp_histogram[threadIdx.x] = 0;
    __syncthreads();

    // Compute shared memory histogram and synchronize
    atomicAdd(&(temp_histogram[input[item]]), 1);
    __syncthreads();

    // Update global histogram
    atomicAdd(&(output[threadIdx.x]), temp_histogram[threadIdx.x]);
}
'''

# compile and setup CUDA code
histogram_gpu = cp.RawKernel(histogram_cuda_code, "histogram")
threads_per_block = 256
grid_size = (int(math.ceil(size / threads_per_block)), 1, 1)
block_size = (threads_per_block, 1, 1)

# check correctness
# histogram(input_cpu, output_cpu)
np_hist = np_histogram(input_cpu, 256)
cp_hist = cp_histogram(input_gpu, 256)
histogram_gpu(grid_size, block_size, (input_gpu, output_gpu))

if np.allclose(np_hist, output_gpu):
    if np.allclose(cp_hist.get(), np_hist):
        print("Correct results!")
else:
    print("Wrong results!")

# measure performance

# print("Timing naive implementation")
# %timeit -n 1 -r 1 histogram(input_cpu, output_cpu)

numpy_hist = benchmark(np_histogram, (input_cpu, 256), n_repeat=1)
print(f"\nNumpy average time: {numpy_hist.to_str()}")

execution_gpu = benchmark(histogram_gpu, (grid_size, block_size, (input_gpu, output_gpu)), n_repeat=10)
execution_cupy_hist = benchmark(cp_histogram, (input_gpu, 256), n_repeat=10)
print(f"\nGPU's raw kernel execution time: {execution_gpu}")
print(f"GPU's cupy hist execution time: {execution_cupy_hist}")

# gpu_avg_time = np.average(execution_gpu.gpu_times)
rawK_avg = np.average(execution_gpu.cpu_times)
cupy_avg = np.average(execution_cupy_hist.cpu_times)
print(f"\ntime spent on CPU for raw kernel: {rawK_avg*1e3} ms")
print(f"times spent on CPU for cupy hist: {cupy_avg*1e3} ms")

rawK_avg_gpu = np.average(execution_gpu.gpu_times)
cupy_avg_gpu = np.average(execution_cupy_hist.gpu_times)
print(f"\ntimes spent on GPU raw kernel: {rawK_avg_gpu*1e3} ms")
print(f"times spent on GPU cupy hist: {cupy_avg_gpu*1e3} ms")