#include <torch/extension.h>
#include "kernels.h" // 헤더 하나로 모든 함수 선언 가져옴

// Add Wrapper
torch::Tensor add_wrapper(torch::Tensor a, torch::Tensor b) {
  // TODO(jpyo0803): 검증 로직 추가
  return custom_add_cuda(a, b);
}

// Matmul Wrapper
torch::Tensor matmul_wrapper(torch::Tensor a, torch::Tensor b) {
  if (a.dim() != 2 || b.dim() != 2) {
    throw std::invalid_argument("Input tensors must be 2-dimensional");
  }
  if (a.size(1) != b.size(1)) {
    throw std::invalid_argument("Inner dimensions must match for matmul");
  }
  return custom_matmul_cuda(a, b);
}

// qk Wrapper
torch::Tensor qk_wrapper(torch::Tensor q, torch::Tensor k) {
  // TODO(jpyo0803): 검증 로직 추가
  return custom_qk_cuda(q, k);
}

// pv Wrapper
torch::Tensor pv_wrapper(torch::Tensor p, torch::Tensor v) {
  // TODO(jpyo0803): 검증 로직 추가
  return custom_pv_cuda(p, v);
}

// 모듈 등록
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("add", &add_wrapper, "Custom Add");
  m.def("matmul", &matmul_wrapper, "Custom MatMul");
  m.def("qk", &qk_wrapper, "Custom QK");
  m.def("pv", &pv_wrapper, "Custom PV");
}