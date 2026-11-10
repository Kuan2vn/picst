import torch
import math

def adjust_learning_rate(method, base_lr, iters, max_iters, power):
    method = str(method).strip().lower()
    progress = min(max(float(iters) / max(float(max_iters), 1.0), 0.0), 1.0)
    if method in ("poly", "polynomial"):
        lr = base_lr * ((1 - progress) ** power)
    elif method in ("cosine", "cos"):
        lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    else:
        raise NotImplementedError("Unsupported LR method: {}".format(method))
    return lr
