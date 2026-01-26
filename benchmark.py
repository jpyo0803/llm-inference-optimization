import torch
from transformers import LlamaConfig

from models.modelling_llama import LlamaForCausalLM

assert torch.cuda.is_available(), "GPU support is required for this benchmark."

# 재현성을 위한 시드 고정
from utils import seed_everything
seed_everything(42)

# 항상 GPU 사용 가정
DEVICE = "cuda"

# Llama 3.2 1B 스펙 정의
config = LlamaConfig(
    vocab_size=128256,   # Llama 3.2 토크나이저 크기
    hidden_size=2048,    # 1B 모델의 hidden size
    intermediate_size=8192, # FFN 확장 크기
    num_hidden_layers=16, # 레이어 수 (1B 기준)
    num_attention_heads=32,
    num_key_value_heads=8,
    max_position_embeddings=2048, # 컨텍스트 길이 (테스트용으로 줄임)
    torch_dtype=torch.float16,
)

# 벤치마크 하이퍼파라미퍼 (부하 조절용)
PROMPT_LEN = 1024  # 입력 시퀀스 길이
GEN_LEN = 512     # 생성할 토큰 수
BATCH_SIZE = 1      # 배치 크기
WARMUP_STEPS = 5   # 워밍업 스텝 수
TEST_STEPS = 10   # 측정 스텝 수

def benchmark():
    # 원본 Llama 모델 생성 및 GPU로 이동
    native_model = LlamaForCausalLM(config).to(DEVICE)
    native_model.eval()
    print("Native Llama model loaded.")

    # 입력 데이터 생성
    input_ids = torch.randint(0, config.vocab_size, (BATCH_SIZE, PROMPT_LEN)).to(DEVICE)

    # 시간 측정 준비
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    print("Warming up...")
    for _ in range(WARMUP_STEPS):
        with torch.no_grad():
            native_model.generate(
                input_ids,
                max_new_tokens=10,
                min_new_tokens=10,
                do_sample=False,
            )
    
    print("Starting benchmark...")
    latencies = []

    for step in range(TEST_STEPS):
        torch.cuda.synchronize()
        start_event.record()

        with torch.no_grad():
            native_model.generate(
                input_ids,
                max_new_tokens=GEN_LEN,
                min_new_tokens=GEN_LEN,
                do_sample=False,
                pad_token_id=config.vocab_size - 1,  # 경고 방지
            )

        end_event.record()
        torch.cuda.synchronize()

        elapsed_time_ms = start_event.elapsed_time(end_event)  # 밀리초 단위
        latencies.append(elapsed_time_ms)
        print(f"Step {step + 1}/{TEST_STEPS}: {elapsed_time_ms:.2f} ms")

    avg_latency_sec = sum(latencies) / len(latencies) / 1000  # 초 단위로 변환
    total_tokens = GEN_LEN
    tps = total_tokens / avg_latency_sec
    print(f"\nAverage latency over {TEST_STEPS} steps: {avg_latency_sec:.2f} seconds")
    print(f"Throughput: {tps:.2f} tokens/second")

if __name__ == "__main__":
    benchmark()