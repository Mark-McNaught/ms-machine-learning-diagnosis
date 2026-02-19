import random
import numpy as np
import torch

def set_seed(seed=42):
    """
    Set random seed for reproducibility across Python, NumPy, and PyTorch.
    
    Args:
        seed (int): Random seed value
    
    Notes:
        - Call this at the start of every notebook/script
        - Some CUDA operations remain non-deterministic even with this
        - For full determinism, training will be slower due to cudnn.deterministic=True
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # PyTorch GPU (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # Make CuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to {seed} for Python, NumPy, and PyTorch")