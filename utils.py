import random
import numpy as np
import torch
import os
import re


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_device(gpu="0,1,2,3,4,5,6,7"):
    if torch.cuda.is_available():
        assert re.fullmatch(r'[0-7](,[0-7])*', gpu) is not None, 'invalid way to specify the gpu numbers'
        os.environ["cuda_visible_devices"] = gpu
        if len(gpu.split(',')) > 1:
            device = torch.device(f'cuda:{gpu.split(",")[0]}')
            torch.backends.cudnn.benchmark = True
        else:
            device = torch.device(f'cuda:{gpu}')
    else:
        device = torch.device('cpu')
    print(f"device: {device}")
    return device

    