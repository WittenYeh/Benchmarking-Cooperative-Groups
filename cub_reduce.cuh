/*
 * @Autor: Witten Yeh
 * @Date: 2024-10-27 00:32:45
 * @LastEditors: Witten Yeh
 * @LastEditTime: 2024-10-27 01:10:39
 * @Description: 
 */

#pragma once

#include <cub/cub.cuh>

namespace benchmark {
namespace cub_reduce {

#include <cub/cub.cuh>
#include <cuda_runtime.h>

__global__ void kernel(int* d_data, int n_valid_item) {
    // Shared memory for CUB
    __shared__ typename cub::BlockReduce<int, 256>::TempStorage temp_storage;

    // Thread index
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize the value to reduce
    int val = (global_tid < n_valid_item) ? d_data[global_tid] : 0;

    // Perform block reduction using CUB
    int sum = cub::BlockReduce<int, 256>(temp_storage).Sum(val);
}

// Launch kernel example
void launch_kernel(int* d_data, int n_valid_item) {
    int blockSize = 256; // Example block size
    int numBlocks = (n_valid_item + blockSize - 1) / blockSize;

    kernel<<<numBlocks, blockSize>>>(d_data, n_valid_item);
}


};  // namespace cub_reduce
};  // namespace benchmark