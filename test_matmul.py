import torch
import custom_backend # 앞서 만든 모듈 import
import time

WARMUP_STEPS = 5
TEST_STEPS = 10

def run_benchmark(matmul_func, a, b):
    # 워밍업
    for _ in range(WARMUP_STEPS):
        _ = matmul_func(a, b)

    # 벤치마크
    latencies = []
    for _ in range(TEST_STEPS):
        torch.cuda.synchronize()
        start_time = time.time()
        
        c = matmul_func(a, b)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        latencies.append((end_time - start_time) * 1000)  # 밀리초 단위

    avg_latency = sum(latencies) / len(latencies)
    return avg_latency, c # 검증을 위해 마지막 결과도 반환

def test():
    print("\n[Custom MatMul Test]\n")
    
    assert torch.cuda.is_available(), "CUDA device is required for this test."
    
    device = torch.device("cuda")
    dtype = torch.float16

    print(f"Target device: {device}")

    # 입력 데이터 준비 (Llama 3.2 1B 모델 흉내)
    # M: Batch Size * Seq Len (예: 1 * 128)
    # K: Input Features (Hidden Size, 예: 2048)
    # N: Output Features (예: 2048)
    M, K, N = 128, 2048, 2048

    a = torch.randn(M, K, device=device, dtype=dtype)
    b = torch.randn(N, K, device=device, dtype=dtype)
    print(f"A shape: [{M}, {K}], B shape: [{N}, {K}]")

    # 커스텀 matmul 함수 벤치마크
    custom_latency, custom_output = run_benchmark(custom_backend.matmul, a, b)

    # 기본 PyTorch matmul 함수 벤치마크
    torch_latency, torch_output = run_benchmark(lambda x, y: torch.matmul(x, y.T), a, b)

    # 결과 검증
    if torch.allclose(custom_output, torch_output, atol=1e-2):
        print("Outputs match within tolerance.")
    else:
        print("Outputs do not match!")

    # 최대 차이 출력
    max_diff = torch.max(torch.abs(custom_output - torch_output)).item()
    print(f"Max difference between outputs: {max_diff:.6f}")

    print(f"\n[Results]")
    print(f"Custom MatMul Avg Latency: {custom_latency:.2f} ms")
    print(f"PyTorch MatMul Avg Latency: {torch_latency:.2f} ms")
    print(f"Speedup: {torch_latency / custom_latency:.2f}x")


if __name__ == "__main__":
    test()