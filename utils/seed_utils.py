import random
import numpy as np
import torch

def set_seeds(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def reset():
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
