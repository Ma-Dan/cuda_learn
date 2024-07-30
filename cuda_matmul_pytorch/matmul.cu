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

    /*cublasSgemm(
        handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k,
        &alpha,
        a, k,
        b, n,
        &beta,
        result, m);*/
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

#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

//来源 https://github.com/ifromeast/cuda_learning
//考虑计算访存比后的第一版分块
__global__ void sgemm_V1(
    float * __restrict__ a, float * __restrict__ b, float * __restrict__ c,
    const int M, const int N, const int K) {

    //以下为考虑到计算和共享内存等综合状况后的经验值
    //如何设置CUDA Kernel中的grid_size和block_size
    //参考 https://mp.weixin.qq.com/s?__biz=MzU5ODY2MTk3Nw==&mid=2247486265&idx=2&sn=2a1c0f6f1fec62fc25c48e74ba61edfe
    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    __shared__ float s_a[BM][BK];
    __shared__ float s_b[BK][BN];

    float r_c[TM][TN] = {0.0};

    //加载时的切分位置和计算时取元素的位置没有关系
    int load_a_smem_m = tid >> 1;  // tid/2, row of s_a
    int load_a_smem_k = (tid & 1) << 2;  // (tid % 2 == 0) ? 0 : 4, col of s_a
    int load_b_smem_k = tid >> 5;   // tid/32, row of s_b
    int load_b_smem_n = (tid & 31) << 2;  // (tid % 32) * 4, col of s_b

    int load_a_gmem_m = by * BM + load_a_smem_m;  // global row of a
    int load_b_gmem_n = bx * BN + load_b_smem_n;  // global col of b

    for (int bk = 0; bk < (K + BK - 1) / BK; bk++) { //K方向切分，每个block都要把a和b的K方向走完
        //16x16个线程分工读取，注意读取和计算时每个线程访问的share内存的位置并不相同
        int load_a_gmem_k = bk * BK + load_a_smem_k;   // global col of a
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
        FLOAT4(s_a[load_a_smem_m][load_a_smem_k]) = FLOAT4(a[load_a_gmem_addr]);
        int load_b_gmem_k = bk * BK + load_b_smem_k;   // global row of b
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
        FLOAT4(s_b[load_b_smem_k][load_b_smem_n]) = FLOAT4(b[load_b_gmem_addr]);

        __syncthreads();

        //全部线程读取完之后开始计算
        #pragma unroll
        for (int k = 0; k < BK; k++) { //这里的循环词序比较特别，把k放在最外层，这样 TM x TN 的r_c矩阵的每个元素都要被更新k次
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                #pragma unroll
                for (int n = 0; n < TN; n++) {
                    int comp_a_smem_m = ty * TM + m;
                    int comp_b_smem_n = tx * TN + n;
                    r_c[m][n] += s_a[comp_a_smem_m][k] * s_b[k][comp_b_smem_n]; //s_a s_b的一次外积的内循环，更新TN个元素，s_a第m行的第k个元素 和 s_b第k行的n个元素 相乘， 更新r_c第m行第n列
                }
                //s_a s_b的一次外积的外循环，更新r_c 的 TM * TN个元素 一次
            }
            //s_a s_b的 BK 次外积，更新r_c 的 TM * TN 个元素 BK 次
        }
        //BK循环 (K + BK - 1) / BK 次，把K方向上 a的一行 和 b的一列 算完
        //每次只能加载BK这么多数据到 shared memory

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int store_c_gmem_m = by * BM + ty * TM + i;
        #pragma unroll
        for (int j = 0; j < TN; j += 4) {
            int store_c_gmem_n = bx * BN + tx * TN + j;
            int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
            FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i][j]);
        }
    }
}

void matmul_v1(float* result, float* a, float* b, int M, int N, int K) {
    const int BM = 128, BN = 128, TM = 8, TN = 8;
    dim3 blockDim(BN / TN, BM / TM);
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

    sgemm_V1<<<gridDim, blockDim>>>(a, b, result, M, N, K);
}