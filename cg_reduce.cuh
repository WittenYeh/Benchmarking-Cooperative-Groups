/*
 * @Autor: Witten Yeh
 * @Date: 2024-10-26 23:26:40
 * @LastEditors: Witten Yeh
 * @LastEditTime: 2024-10-27 01:11:40
 * @Description: 
 */

#include <cuda_runtime.h>
#include <cooperative_groups.h>

#pragma once

namespace benchmark {
namespace cg_reduce {

namespace cg = cooperative_groups;

__device__ int reduce_sum(cg::thread_group g, int *temp, int val) {
    int lane = g.thread_rank();

    // Each iteration halves the number of active threads
    // Each thread adds its partial sum[i] to sum[lane+i]
    for (int i = g.size() / 2; i > 0; i /= 2)
    {
        temp[lane] = val;
        g.sync(); // wait for all threads to store
        if(lane < i) { 
            val += temp[lane + i];
        }
        g.sync(); // wait for all threads to load
    }
    return val; // note: only thread 0 will return full sum
}

__global__ void kernel(int* d_data, int n_valid_item) {

    __shared__ int temp [256]; // Declare shared memory

    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;

    int val = (global_tid < n_valid_item) ? d_data[global_tid] : 0;
    cg::thread_group g = cg::this_thread_block();

    val = reduce_sum(g, temp, val);
}

// Kernel launch example
void launch_kernel(int* d_data, int n_valid_item) {
    int blockSize = 256; // Example block size
    int numBlocks = (n_valid_item + blockSize - 1) / blockSize;
    kernel<<<numBlocks, blockSize>>>(d_data, n_valid_item);
}

};
};  // namespace benchmark