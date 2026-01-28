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

torch::Tensor custom_qk_cuda(torch::Tensor q, torch::Tensor k) {
  // q: [Batch, Head, Q Len, Head Dim]
  // k: [Batch, Head, K Len, Head Dim]
  
  auto batch_size = q.size(0);
  auto num_heads = q.size(1);
  auto q_len = q.size(2);
  auto k_len = k.size(2);
  auto head_dim = q.size(3);

  // 전체 배치 개수 = Batch * Head
  int batch_count = batch_size * num_heads;
  
  // 결과 텐서 생성 [Batch, Head, Seq, Seq]
  auto options = torch::TensorOptions().dtype(q.dtype()).device(q.device());
  auto scores = torch::empty({batch_size, num_heads, q_len, k_len}, options);

  cublasHandle_t handle = get_cublas_handle();

  // Stride 설정 (메모리에서 다음 행렬로 넘어가기 위한 거리)
  // 하나의 행렬(Seq x Dim) 크기만큼 건너뛰어야 함
  long long int strideA = q_len * head_dim; // A의 한 헤드 크기
  long long int strideB = k_len * head_dim; // B의 한 헤드 크기
  long long int strideC = q_len * k_len;  // 결과(C)의 한 헤드 크기

  constexpr float kAlpha = 1.0f;
  constexpr float kBeta = 0.0f;

  /*
    cuBLAS 호출

    쉽게 설명하자면 하나의 2D 행렬 곱셈 [m x k] * [k x n] 을
    한번의 kernel 호출로 batch_count 만큼 수행하는 것입니다.
  */
  
  CUBLAS_CHECK(cublasGemmStridedBatchedEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N, // K는 Transpose해서 곱함
        k_len, q_len, head_dim, // m, n, k
        &kAlpha,
        k.data_ptr(), CUDA_R_16F, head_dim, strideB, // B 행렬 자리 (K)
        q.data_ptr(), CUDA_R_16F, head_dim, strideA, // A 행렬 자리 (Q)
        &kBeta,
        scores.data_ptr(), CUDA_R_16F, k_len, strideC, // C 행렬 자리
        batch_count, // 전체 배치 수
        CUDA_R_32F,  // FP32 Accumulation
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
  ));

  return scores;
}

torch::Tensor custom_pv_cuda(torch::Tensor p, torch::Tensor v) {
  // p: [Batch, Head, Q Len, V Len]
  // v: [Batch, Head, V Len, Head Dim]
  
  auto batch_size = p.size(0);
  auto num_heads = p.size(1);
  auto q_len = p.size(2);
  auto v_len = p.size(3);
  auto head_dim = v.size(3);

  // 전체 배치 개수 = Batch * Head
  int batch_count = batch_size * num_heads;
  
  // 결과 텐서 생성 [Batch, Head, Seq, Head Dim]
  auto options = torch::TensorOptions().dtype(p.dtype()).device(p.device());
  auto output = torch::empty({batch_size, num_heads, q_len, head_dim}, options);

  cublasHandle_t handle = get_cublas_handle();

  // Stride 설정 (메모리에서 다음 행렬로 넘어가기 위한 거리)
  long long int strideA = q_len * v_len; // A의 한 헤드 크기
  long long int strideB = v_len * head_dim; // B의 한 헤드 크기
  long long int strideC = q_len * head_dim;  // 결과(C)의 한 헤드 크기

  constexpr float kAlpha = 1.0f;
  constexpr float kBeta = 0.0f;

  CUBLAS_CHECK(cublasGemmStridedBatchedEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        head_dim, q_len, v_len,
        &kAlpha,
        v.data_ptr(), CUDA_R_16F, head_dim, strideB,
        p.data_ptr(), CUDA_R_16F, v_len, strideA,
        &kBeta,
        output.data_ptr(), CUDA_R_16F, head_dim, strideC,
        batch_count,
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
  ));

  return output;
}