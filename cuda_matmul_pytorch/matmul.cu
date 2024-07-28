#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>
#include <cublas_v2.h>

cublasHandle_t handle;

void createCublas() {
    cublasCreate(&handle);
}

void destroyCublas() {
    cublasDestroy(handle);
}

void matmul_cublas(float* result, float* a, float* b, int m, int n, int k) {
    // Calculate with Cublas
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
        &alpha,
        b, n,
        a, k,
        &beta,
        result, n);
}

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

__global__ void naiveSgemm(
    float * __restrict__ a, float * __restrict__ b, float * __restrict__ c,
    const int M, const int N, const int K) {

    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m < M && n < N) {
        float psum = 0.0;
        #pragma unroll
        for (int k = 0; k < K; k++) {
            psum += a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];
        }
        c[OFFSET(m, n, N)] = psum;
    }
}

void matmul_naive(float* result, float* a, float* b, int M, int N, int K) {
    const int BM = 32, BN = 32;

    dim3 blockDim(BN, BM);
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

    naiveSgemm<<<gridDim, blockDim>>>(a, b, result, M, N, K);
}
