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