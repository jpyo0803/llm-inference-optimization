import torch
import torch.nn as nn
import custom_backend # C++ Extension 모듈

class CustomLinear(nn.Linear):
  def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
    super().__init__(in_features, out_features, bias, device, dtype)

  def forward(self, input):
    # input: [batch_size, seq_len, in_features]

    # 메모리 연속성 보장
    if not input.is_contiguous():
        input = input.contiguous()

    # 3D -> 2D Flatten
    # [Batch, Seq, In] -> [Batch * Seq, In]
    orig_shape = input.shape
    input_2d = input.view(-1, self.in_features)

    # Custom Backend MatMul 호출
    output_2d = custom_backend.matmul(input_2d, self.weight)

    # 원래 차원으로 복구
    output_shape = list(orig_shape)
    output_shape[-1] = self.out_features
    output = output_2d.view(*output_shape)

    # Bias 더하기
    if self.bias is not None:
        output = output + self.bias
        
    return output