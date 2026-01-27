import torch
import custom_backend # 앞서 만든 모듈 import

def test():
    print("\n--- Custom MatMul Test (FP16) ---")
    
    # 1. 디바이스 설정
    if not torch.cuda.is_available():
        print("Error: CUDA device is required for this test.")
        return
    
    device = torch.device("cuda")
    dtype = torch.float16
    print(f"Testing on device: {device}")

    # 입력 데이터 준비 (Llama 3.2 1B 모델 흉내)
    # M: Batch Size * Seq Len (예: 1 * 128)
    # K: Input Features (Hidden Size, 예: 2048)
    # N: Output Features (예: 2048)
    M, K, N = 128, 2048, 2048
    
    print(f"Shapes -> Input: [{M}, {K}], Weight: [{N}, {K}]")

    # 입력 데이터 (Row-Major)
    x = torch.randn(M, K, device=device, dtype=dtype)
    
    # 가중치 데이터 (nn.Linear는 [Out, In] 형태로 저장됨 -> [N, K])
    w = torch.randn(N, K, device=device, dtype=dtype)

    # 내 C++ 함수 호출
    # 커널 내부에서 W^T를 수행하므로 그대로 넘김
    print("Running Custom Backend...", end="")
    torch.cuda.synchronize() # 정확한 타이밍을 위해 대기
    output_custom = custom_backend.matmul(x, w)
    torch.cuda.synchronize()
    print(" Done.")

    # PyTorch Native 연산 (비교군)
    # 우리가 만든 커널은 Linear Layer용이므로 (X @ W.T)와 같음
    print("Running PyTorch Native...", end="")
    torch.cuda.synchronize()
    output_native = torch.matmul(x, w.t()) # w를 Transpose 해줘야 함!
    torch.cuda.synchronize()
    print(" Done.")

    # 결과 검증
    # FP16은 정밀도가 낮아 오차가 조금 있을 수 있으므로 rtol, atol을 넉넉히 줌
    print(f"\nCustom Output Shape: {output_custom.shape}")
    print(f"Native Output Shape: {output_native.shape}")
    
    # 최대 오차 계산
    diff = (output_custom - output_native).abs().max().item()
    print(f"Max Difference: {diff:.6f}")

    if torch.allclose(output_custom, output_native, rtol=1e-2, atol=1e-2):
        print("\nSuccess! Custom MatMul works correctly.")
    else:
        print("\nFail! Results mismatch too much.")
        # 디버깅용 출력
        print("First 5 values (Custom):", output_custom[0, :5])
        print("First 5 values (Native):", output_native[0, :5])

if __name__ == "__main__":
    test()