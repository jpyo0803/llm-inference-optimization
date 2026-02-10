from locust import HttpUser, task, between
import random

# 테스트에 사용할 예제 프롬프트들
PROMPTS = [
    "Explain quantum computing in simple terms.",
    "Write a poem about the sea.",
    "What is the capital of France?",
    "Python code for fibonacci sequence.",
    "Translate 'Hello' to Spanish.",
]

class LLMUser(HttpUser):
    wait_time = between(0.5, 2.0) # 0.5초에서 2초 사이 랜덤 대기후 재요청

    @task
    def generate_text(self):
        prompt = random.choice(PROMPTS)
        payload = {"prompt": prompt}
        
        # POST 요청 전청
        with self.client.post("/generate", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")