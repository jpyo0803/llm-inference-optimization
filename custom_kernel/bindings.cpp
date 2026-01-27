#include <torch/extension.h>
#include "kernels.h" // 헤더 하나로 모든 함수 선언 가져옴

// Matmul Wrapper
torch::Tensor matmul_wrapper(torch::Tensor a, torch::Tensor b) {
  // TODO(jpyo0803): 검증 로직 추가
  return custom_matmul_cuda(a, b);
}

// Add Wrapper
torch::Tensor add_wrapper(torch::Tensor a, torch::Tensor b) {
  // TODO(jpyo0803): 검증 로직 추가
  return custom_add_cuda(a, b);
}

// 모듈 등록
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matmul", &matmul_wrapper, "Custom MatMul");
  m.def("add", &add_wrapper, "Custom Add");
}