import torch
import torch.nn as nn

def init_weights(m, bias=0.1):

    """Initialize weights for nn.Module (Conv2d, Conv2dT, BatchNorm2d, Linear)"""

    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, bias)

    if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        nn.init.uniform_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
        
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
