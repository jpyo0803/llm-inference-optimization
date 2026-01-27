#pragma once
#include <torch/extension.h>

// Extension 헤더 선언
torch::Tensor custom_add_cuda(torch::Tensor a, torch::Tensor b);

torch::Tensor custom_matmul_cuda(torch::Tensor a, torch::Tensor b);
