import torch
import simple_extension # 앞서 만든 모듈 import

def test():
    print("--- Simple Addition Test ---")
    
    # 데이터 준비 (GPU가 있으면 GPU로 테스트)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")
    
    x = torch.tensor([1.0, 2.0, 3.0], device=device)
    y = torch.tensor([4.0, 5.0, 6.0], device=device)
    
    # 내 C++ 함수 호출
    output_custom = simple_extension.simple_add(x, y)
    
    # PyTorch 원래 덧셈
    output_native = x + y
    
    print(f"Input X: {x}")
    print(f"Input Y: {y}")
    print(f"Custom Output: {output_custom}")
    print(f"Native Output: {output_native}")
    
    # 4. 검증
    if torch.allclose(output_custom, output_native):
        print("\nSuccess! Python <-> C++ Connection is working.")
    else:
        print("\nFail! Values mismatch.")

if __name__ == "__main__":
    test()