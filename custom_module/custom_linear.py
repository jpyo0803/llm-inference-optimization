import torch
import torch.nn as nn
import custom_backend # C++ Extension 모듈

class CustomLinear(nn.Module):
  def __init__(self, in_features, out_features, bias=True, dtype=torch.float16):
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.dtype = dtype

    # 재현성을 위해서 시드를 고정하고 다음과 같이 초기화합니다.
    self.weight = nn.Parameter(torch.randn(out_features, in_features, dtype=dtype))

    if bias:
        self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype))
    else:
        self.register_parameter('bias', None)

  def forward(self, x):
    # x: [batch_size, seq_len, in_features]

    # 입력을 2D 텐서로 변환합니다. 
    # [batch_size * seq_len, in_features] -> [batch_size * seq_len, out_features]
    orig_shape = x.shape
    x_2d = x.view(-1, self.in_features)

    output_2d = custom_backend.matmul(x_2d, self.weight)

    new_shape = list(orig_shape)
    new_shape[-1] = self.out_features
    output = output_2d.view(*new_shape)

    if self.bias is not None:
        output += self.bias
    
    return output