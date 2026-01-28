#pragma once
#include <torch/extension.h>

// Extension 헤더 선언
torch::Tensor custom_add_cuda(torch::Tensor a, torch::Tensor b);

torch::Tensor custom_matmul_cuda(torch::Tensor a, torch::Tensor b);

torch::Tensor custom_qk_cuda(torch::Tensor q, torch::Tensor k);

torch::Tensor custom_pv_cuda(torch::Tensor p, torch::Tensor v);
