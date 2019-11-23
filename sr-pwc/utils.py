import torch
import torch.nn.functional as F

import math

def psnr(x, y, maxi=2.):
    mse = F.mse_loss(x, y)
    return 10 * math.log10(maxi**2 / mse)

def avg_psnr(x, y, maxi=2.):
    n = 1
    dim = y.shape
    if len(dim) > 3:
        n = dim[0]

    p = 0.
    for i in range(n):
        p += psnr(x[i, :, :, :], y[i, :, :, :], maxi=maxi)

    return p / n

