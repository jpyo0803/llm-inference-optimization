import unittest
import torch
from custom_module.custom_matmul import custom_matmul, custom_qk, custom_pv

class TestCustomKernels(unittest.TestCase):
    def test_matmul(self):
        a = torch.randn(3, 5, device='cuda', dtype=torch.float16)
        b = torch.randn(5, 4, device='cuda', dtype=torch.float16)
        expected = torch.matmul(a, b)
        result = custom_matmul(a, b.transpose(0, 1))
        self.assertTrue(torch.allclose(expected, result), "MatMul kernel failed")

    def test_qk(self):
        q = torch.randn(2, 4, 8, 16, device='cuda', dtype=torch.float16)
        k = torch.randn(2, 4, 16, 32, device='cuda', dtype=torch.float16)
        expected = torch.matmul(q, k)
        result = custom_qk(q, k.transpose(-2, -1))
        self.assertTrue(torch.allclose(expected, result), "QK kernel failed")

    def test_pv(self):
        p = torch.randn(2, 4, 8, 16, device='cuda', dtype=torch.float16)
        v = torch.randn(2, 4, 16, 32, device='cuda', dtype=torch.float16)
        expected = torch.matmul(p, v)
        result = custom_pv(p, v)
        self.assertTrue(torch.allclose(expected, result), "PV kernel failed")

if __name__ == "__main__":
    unittest.main()