#include <pybind11/pybind11.h>
#include <torch/extension.h>

void createCublas();
void destroyCublas();
void matmul_cublas(float* result, float* a, float* b, int m, int n, int k);
void matmul_naive(float* result, float* a, float* b, int M, int N, int K);

namespace ops {
    void matmul(const at::Tensor &a, const at::Tensor &b, const at::Tensor &result) {
        auto aSizes = a.sizes();
        auto bSizes = b.sizes();

        int m = aSizes[0];
        int n = bSizes[1];
        int k = aSizes[1];

        //cublas
        /*createCublas();
        matmul_cublas((float*)result.data_ptr(), (float*)a.data_ptr(), (float*)b.data_ptr(), m, n, k);
        destroyCublas();*/

        //naive
        matmul_naive((float*)result.data_ptr(), (float*)a.data_ptr(), (float*)b.data_ptr(), m, n, k);
    }
} // namespace ops

PYBIND11_MODULE(cuda_learn, m) {
m.doc() = "cuda learn pybind11 interfaces";
m.def("matmul", &ops::matmul, "");
}
