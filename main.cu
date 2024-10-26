/*
 * @Autor: Witten Yeh
 * @Date: 2024-10-27 00:45:17
 * @LastEditors: Witten Yeh
 * @LastEditTime: 2024-10-27 01:07:39
 * @Description: 
 */

#include "cg_reduce.cuh"
#include "cub_reduce.cuh"

int main() {
    int n_valid_item = 1 << 16; 
    int* h_data = new int[n_valid_item];

    // Initialize host data
    for (int i = 0; i < n_valid_item; i++) {
        h_data[i] = i; // For simplicity, let's fill with ones
    }

    // Allocate device memory
    int* d_data;
    cudaMalloc((void**)&d_data, n_valid_item * sizeof(int));
    cudaMemcpy(d_data, h_data, n_valid_item * sizeof(int), cudaMemcpyHostToDevice);

    // Start timing
    cudaEvent_t start1, stop1;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);

    cudaEventRecord(start1);

     // Launch the kernel
    for (int i = 0; i < 1 << 20; ++i)
        benchmark::cg_reduce::launch_kernel(d_data, n_valid_item);

    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1);

    // Calculate elapsed time
    float milliseconds1 = 0;
    cudaEventElapsedTime(&milliseconds1, start1, stop1);
    std::cout << "cg_reduce kernel execution time: " << milliseconds1 << " ms" << std::endl;

    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);

    cudaEventRecord(start2);

    // Launch the kernel
    for (int i = 0; i < 1 << 20; ++i)
        benchmark::cub_reduce::launch_kernel(d_data, n_valid_item);

    cudaEventRecord(stop2);
    cudaEventSynchronize(stop2);

    // Calculate elapsed time
    float milliseconds2 = 0;
    cudaEventElapsedTime(&milliseconds2, start2, stop2);
    std::cout << "cub_reduce kernel execution time: " << milliseconds2 << " ms" << std::endl;
    
}