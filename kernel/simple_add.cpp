#include <torch/extension.h>

torch::Tensor simple_add(torch::Tensor a, torch::Tensor b) {
  TORCH_CHECK(a.sizes() == b.sizes(), "Input tensors must have the same size");
  return a + b;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("simple_add", &simple_add, "A simple addition of two tensors");
}