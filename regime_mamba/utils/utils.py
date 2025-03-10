import numpy as np
import torch
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

def set_seed(seed=42):
    """재현성을 위한 시드 설정"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False