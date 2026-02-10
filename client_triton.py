import numpy as np
import tritonclient.http as httpclient
from transformers import AutoTokenizer

# 설정 (모델명은 config.pbtxt의 'name'과 일치해야 함)
TRITON_URL = "localhost:8000"
MODEL_NAME = "llama_onnx"     
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

def test_inference():
    print(f"Connecting to Triton at {TRITON_URL}...")
    try:
        client = httpclient.InferenceServerClient(url=TRITON_URL)
        if not client.is_server_live():
            print("Server is NOT live.")
            return
        print("Server is Live.")
    except Exception as e:
        print(f"Connection Failed: {e}")
        return

    # 토크나이저 로드
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # 입력 데이터 준비
    prompt = "Hello, my name is"
    print(f"Prompt: '{prompt}'")
    
    # Triton은 INT64 형태의 입력 기대
    inputs = tokenizer(prompt, return_tensors="np")
    input_ids = inputs["input_ids"].astype(np.int64)
    attention_mask = inputs["attention_mask"].astype(np.int64)
    
    # 입력 객체 생성
    triton_inputs = [
        httpclient.InferInput("input_ids", input_ids.shape, "INT64"),
        httpclient.InferInput("attention_mask", attention_mask.shape, "INT64")
    ]
    
    triton_inputs[0].set_data_from_numpy(input_ids)
    triton_inputs[1].set_data_from_numpy(attention_mask)
    
    # 출력 객체 생성
    triton_outputs = [
        httpclient.InferRequestedOutput("logits")
    ]

    # 추론 요청
    print("Sending request to Triton...")
    response = client.infer(
        model_name=MODEL_NAME,
        inputs=triton_inputs,
        outputs=triton_outputs
    )

    # 결과 확인
    logits = response.as_numpy("logits")
    print(f"Inference Success! Logits Shape: {logits.shape}")
    
    # 다음 단어 예측
    next_token_id = np.argmax(logits[0, -1, :])
    next_token = tokenizer.decode(next_token_id)
    print(f"Predicted Next Token: '{next_token}'")

if __name__ == "__main__":
    test_inference()