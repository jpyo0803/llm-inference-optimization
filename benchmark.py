import torch
from transformers import LlamaConfig

from model.modelling_llama import LlamaForCausalLM
from model.custom_modelling_llama import LlamaForCausalLM as CustomLlamaForCausalLM

assert torch.cuda.is_available(), "GPU support is required for this benchmark."

# 재현성을 위한 시드 고정
from utils import seed_everything, clean_gpu_memory
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

# Default attention 구현인 eager 모드 사용
config._attn_implementation = "eager"

# 벤치마크 하이퍼파라미퍼 (부하 조절용)
PROMPT_LEN = 1024  # 입력 시퀀스 길이
GEN_LEN = 512     # 생성할 토큰 수
BATCH_SIZE = 2      # 배치 크기
WARMUP_STEPS = 1   # 워밍업 스텝 수
TEST_STEPS = 3   # 측정 스텝 수

dummy_input = torch.randint(0, config.vocab_size, (BATCH_SIZE, PROMPT_LEN)).to(DEVICE)
attention_mask = torch.ones_like(dummy_input).to(DEVICE)

def benchmark(model):
    # 원본 Llama 모델 생성 및 GPU로 이동
    model = model.to(DEVICE).half()
    model.eval()

    print("Warming up...")
    for _ in range(WARMUP_STEPS):
        with torch.no_grad():
            model.generate(
                dummy_input,
                attention_mask=attention_mask,
                max_new_tokens=10,
                min_new_tokens=10,
                do_sample=False,
                pad_token_id=config.vocab_size - 1,  # 경고 방지
            )
    
    print("Starting benchmark...")
    latencies = []

    # 시간 측정 준비
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    first_token_id = -1

    for step in range(TEST_STEPS):
        torch.cuda.synchronize()
        start_event.record()

        with torch.no_grad():
            output = model.generate(
                dummy_input,
                attention_mask=attention_mask,
                max_new_tokens=GEN_LEN,
                min_new_tokens=GEN_LEN,
                do_sample=False,
                pad_token_id=config.vocab_size - 1,  # 경고 방지
            )

        end_event.record()
        torch.cuda.synchronize()

        if step == 0:
            last_token_id = output[0, -1].item()

        elapsed_time_ms = start_event.elapsed_time(end_event)  # 밀리초 단위
        latencies.append(elapsed_time_ms)
        print(f"Step {step + 1}/{TEST_STEPS}: {elapsed_time_ms:.2f} ms")

    avg_latency_sec = sum(latencies) / len(latencies) / 1000  # 초 단위로 변환

    return avg_latency_sec, last_token_id  # 평균 지연 시간과 마지막 생성 토큰 ID 반환

def main():
    print("[Native Llama Model]")
    native_model = LlamaForCausalLM(config)
    state_dict = native_model.state_dict()

    # 입력 데이터 생성
    dummy_input = torch.randint(0, config.vocab_size, (BATCH_SIZE, PROMPT_LEN)).to(DEVICE)

    native_avg_latency, native_last_token = benchmark(native_model)
    del native_model
    clean_gpu_memory()

    print("[Custom Llama Model]")
    custom_model = CustomLlamaForCausalLM(config)
    msg = custom_model.load_state_dict(state_dict, strict=False)
    print(f"Weight Loading Result: {msg}")

    custom_avg_latency, custom_last_token = benchmark(custom_model)
    del custom_model
    clean_gpu_memory()

    if native_last_token == custom_last_token:
        print(f"Last generated token matches: {native_last_token}")
    else:
        print(f"Last generated token mismatch! Native: {native_last_token}, Custom: {custom_last_token}")

    print("\n[Benchmark Results]")
    print(f"Native Model Avg Latency: {native_avg_latency:.2f} seconds")
    print(f"Custom Model Avg Latency: {custom_avg_latency:.2f} seconds")
    print(f"Speedup: {native_avg_latency / custom_avg_latency:.2f}x")

if __name__ == "__main__":
    main()