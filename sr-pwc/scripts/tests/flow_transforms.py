import sys
import os
import pprint
import pdb

import torch

import torchvision
import torchvision.transforms as transforms

sys.path.append('../..')

from data_utils import Sintel

from data_utils import RandomHorizontalFlip
from flow_utils import ScaleFlow, ResizeFlow, ToTensor, ToRGBImage, ToPILImage

dataset = Sintel('/data1/data/sintel', split='training', flow_scale=1, pyramid_levels=[0, 1, 2])

image1, image2, flow = dataset.__getitem__(10)

# scaling
scale_transform = ScaleFlow(10)
scaled_flow = scale_transform(flow[0])

# to tensor
tensor_transform = ToTensor()
tensor_flow = tensor_transform(flow[0])

# to rgb image
rgb_transform = ToRGBImage()
rgb_flow = rgb_transform(flow[0])

# to pil image
pil_transform = ToPILImage()
pil_flow = pil_transform(tensor_flow)

# resize
flip_transform = RandomHorizontalFlip(1)
image1_flip = flip_transform(image1)
image2_flip = flip_transform(image2)
flow_flip = pil_transform(tensor_transform(flip_transform(flow[0])))


pdb.set_trace()
