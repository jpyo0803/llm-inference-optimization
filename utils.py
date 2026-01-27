import torch
import random
import numpy as np
import os

def seed_everything(seed: int = 42):
    # Python 기본 시드 고정
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # NumPy 시드 고정
    np.random.seed(seed)
    
    # PyTorch (CPU & GPU) 시드 고정
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 멀티 GPU 사용 시
    
    # 연산 결정론적(Deterministic) 설정
    # 속도가 약간 느려질 수 있지만, 같은 입력에 대해 무조건 같은 결과를 보장합니다.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

import gc

def clean_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()