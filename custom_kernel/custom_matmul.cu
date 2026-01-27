#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#define CUBLAS_CHECK(err) \
  if (err != CUBLAS_STATUS_SUCCESS) { \
      printf("cuBLAS Error: %d at %s:%d\n", err, __FILE__, __LINE__); \
      exit(1); \
  }

// 성능을 위해 cuBLAS 핸들을 전역으로 공유 
cublasHandle_t get_cublas_handle() {
  static cublasHandle_t handle = nullptr;
  if (!handle) {
      cublasCreate(&handle);
  }
  return handle;
}

torch::Tensor custom_matmul_cuda(torch::Tensor a, torch::Tensor b) {
  // a: [M, K]
  // b: [N, K], nn.Linear stores as out_features x in_features
  // c: [M, N]

  auto m = a.size(0);
  auto k = a.size(1);
  auto n = b.size(0);

  auto options = torch::TensorOptions().dtype(a.dtype()).device(a.device());
  auto c = torch::empty({m, n}, options);

  cublasHandle_t handle = get_cublas_handle();

  constexpr float kAlpha = 1.0f;
  constexpr float kBeta = 0.0f;

  /*
    cuBLAS는 column-major 방식을 사용하므로,
    행렬 곱셈 C = A * B 를 수행할 때,
    실제로는 C^T = B^T * A^T 로 계산해야 합니다. B^T * A^T = (A * B)^T
  */
  CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        n, m, k,
        &kAlpha,
        b.data_ptr(), CUDA_R_16F, k,
        a.data_ptr(), CUDA_R_16F, k,
        &kBeta,
        c.data_ptr(), CUDA_R_16F, n,
        // 입력은 FP16으로 받되 intermediate 연산은 FP32로 수행, 출력은 FP16
        CUDA_R_32F, 
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
  ));

  return c;
}