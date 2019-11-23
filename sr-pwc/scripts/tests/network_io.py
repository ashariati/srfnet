import sys
import os
import pprint

import torch

sys.path.append('../..')

from networks import DenseNet
from networks import PWCNet
from networks import SRResNet
from networks import SRPWCNet

from flow_utils import RobustLoss
from flow_utils import MaskedRobustLoss

import pdb

torch.manual_seed(3607)

# DenseNet
# X = torch.randn(1, 81, 15, 4)

# out_channels = [128, 128, 96, 64, 32]
# net = DenseNet(81, out_channels)
# pprint.pprint(net)
# net(X)
 
# PWCNet
# X = torch.randn(1, 6, 384, 768).cuda()
# X = torch.randn(1, 6, 96, 192).cuda()
# net = PWCNet().cuda()

# SRResNet
# X = torch.randn(1, 3, 56, 20)
# net = SRResNet()
# 
# pprint.pprint(net)
# y, c1, c2, c3 = net(X)
# 
# print(y.shape)
# print(c1.shape)
# print(c2.shape)
# print(c3.shape)


# SRPWCNet / losses
# criterion = RobustLoss(0.01, 0.4)
criterion = MaskedRobustLoss(0.01, 0.4)
X1 = torch.randn(4, 3, 24, 48).cuda()
X2 = torch.randn(4, 3, 24, 48).cuda()
W = torch.randn(4, 2, 96, 192).cuda()
mask = torch.randint(0, 2, (4, 1, 96, 192)).to(dtype=torch.uint8).cuda()
net = SRPWCNet(SRResNet().cuda(), PWCNet().cuda()).cuda()
flows = net(X1, X2)

# loss = criterion(flows[0], W)
loss = criterion(flows[0], W, mask)

pdb.set_trace()

