import numpy as np
import torch
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False  # Prevent minus sign rendering issues

def set_seed(seed=42):
    """Set seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False