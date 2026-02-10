import torch
import os
import sys
from transformers import AutoConfig

sys.path.append(os.getcwd())
from model.modelling_llama import LlamaForCausalLM

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
OUTPUT_PATH = "model_repository/llama_onnx/1/model.onnx"

class LlamaWrapper(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = LlamaForCausalLM(config).half()

    def forward(self, input_ids, attention_mask):
        # use_cache=False: ONNX 변환 시 KV Cache 객체(DynamicCache)가 생성되지 않도록 함
        # TODO(jpyo0803): KV Cache 미사용시 매번 전체 시퀀스를 다시 처리해야 하므로 느림. 추후 KV Cache 
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,  # Cache 끄기
            return_dict=True
        )
        return outputs.logits

def export_onnx():
    print(f"Loading Configuration from {MODEL_ID}...")
    config = AutoConfig.from_pretrained(MODEL_ID)

    model = LlamaWrapper(config)
    model.eval()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    dummy_input = torch.randint(0, 1000, (1, 10), dtype=torch.long)
    dummy_mask = torch.ones((1, 10), dtype=torch.long)

    # Batch size와 시퀀스 길이는 variable함을 설정 
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size", 1: "sequence_length"}
    }

    print(f"Exporting model to ONNX at {OUTPUT_PATH}...")
    torch.onnx.export(
        model,
        (dummy_input, dummy_mask),
        OUTPUT_PATH,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=18,
        do_constant_folding=True,
    )

    print("ONNX export completed.")

if __name__ == "__main__":
    export_onnx()