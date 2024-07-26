#include <pybind11/pybind11.h>
#include <torch/extension.h>


namespace ops {
    void matmul(const at::Tensor &a, const at::Tensor &b, const at::Tensor &result) {

    }
} // namespace ops

PYBIND11_MODULE(cuda_learn, m) {
m.doc() = "cuda learn pybind11 interfaces";
m.def("matmul", &ops::matmul, "");
}
