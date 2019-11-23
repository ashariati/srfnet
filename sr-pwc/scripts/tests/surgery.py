import sys
import os
import pprint

import numpy as np

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils

import tensorboardX
from tensorboardX import SummaryWriter

sys.path.append('../..')

import data_utils
from data_utils import Sintel
from data_utils import FlyingChairs

import flow_utils

import networks
from networks import PWCNet
from networks import SRResNet
from networks import SRPWCNet

import pdb


pwc_model = PWCNet().cuda()
checkpoint_file = os.path.join(os.getcwd(), '../states', 'flying_chairs','pwc_70.pkl')
pwc_model.load_state_dict(torch.load(checkpoint_file))

srresnet_model = SRResNet().cuda()
checkpoint_file = os.path.join(os.getcwd(), '../states', 'flying_chairs','srresnet_95.pkl')
srresnet_model.load_state_dict(torch.load(checkpoint_file))

srpwc_model = SRPWCNet(srresnet_model, pwc_model)

X1 = torch.randn(1, 3, 16, 16).cuda()
X2 = torch.randn(1, 3, 16, 16).cuda()

Y = srpwc_model(X1, X2)

pdb.set_trace()

