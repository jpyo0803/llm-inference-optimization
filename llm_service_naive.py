import sys
import os
import asyncio
import time
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from transformers import AutoConfig, AutoTokenizer

from model.modelling_llama import LlamaForCausalLM

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct" # 구조(Config)만 빌려옴
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

assert DEVICE == "cuda", "GPU is required to run this service."

model = None
tokenizer = None
queue = asyncio.Queue()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    print(f"Loading Configuration from {MODEL_ID} (No Weights)...")
    
    # Config & Tokenizer 로드 (구조와 전처리기만 가져옴)
    config = AutoConfig.from_pretrained(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # 모델 초기화 (Random Weights)
    model = LlamaForCausalLM(config)
    
    model.half()

    # GPU 이동 및 FP16 변환
    model.to(DEVICE)

    model.eval()
    
    # 워커 시작
    asyncio.create_task(process_queue())
    print("Model Initialized with Random Weights & Worker Started.")
    yield

app = FastAPI(lifespan=lifespan)

class RequestData(BaseModel):
    prompt: str

async def process_queue():
    """요청 처리 루프"""
    while True:
        req_id, prompt, future = await queue.get()
        try:
            # Tokenization (연산 부하에 포함)
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=50,  # 최대 길이
                    min_new_tokens=50,  # 최소 길이
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=False     # Greedy하게 진행 (연산량 고정)
                )
            
            # Decoding (결과는 걁뀕 같은 외계어겠지만 길이는 맞음)
            result_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            future.set_result(result_text)
        except Exception as e:
            future.set_exception(e)
        finally:
            queue.task_done()

@app.post("/generate")
async def generate(data: RequestData):
    future = asyncio.get_running_loop().create_future()
    await queue.put((time.time(), data.prompt, future))
    return {"text": await future}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=30000)
