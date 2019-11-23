import sys
import os
import pprint
import math

import numpy as np

import PIL

import torch
from torch import nn

import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils

sys.path.append('..')

import data_utils
from data_utils import KITTIDerot

import flow_utils
import test_models

import pdb

# parameters
window = 50
sequence = 7
frame = 0
# sequence = 8
# frame = 67
# sequence = 4
# frame = 10

imageviz_transform = transforms.Compose([transforms.ToPILImage(),
    transforms.Resize((80, 224), interpolation=PIL.Image.NEAREST),
    transforms.ToTensor()])
flowviz_transform = transforms.Compose([flow_utils.ToFlow(),
    flow_utils.ToRGBImage(),
    transforms.ToTensor()])

image_dir = os.path.join(os.getcwd(), 'flow_gts')
if not os.path.isdir(image_dir):
    os.mkdir(image_dir)

for i in range(frame, frame + window):

    # load new dataset
    io_dataset = KITTIDerot('/mnt/tmp/data/kitti_derot/coarsesim_test', sequences=[sequence], 
            transform=None, input_scale=None, frame_offset=[1],
            pyramid_levels=[2, 3, 4], data_augmentation=False, fflip=-1, return_id=True)
    
    _, _, flow_gt, _, _, _, _ = io_dataset.__getitem__(i)
    
    flow_gt_name = os.path.join(image_dir, "flow_gt_%02d.png" % i)
    vutils.save_image(flowviz_transform(flow_gt[0]), flow_gt_name, nrow=1)

