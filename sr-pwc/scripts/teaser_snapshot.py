import sys
import os
import pprint
import math

import numpy as np

import PIL
import cv2

import torch
from torch import nn

import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torchvision.utils as vutils

sys.path.append('..')

import data_utils
from data_utils import KITTIDerot

import flow_utils
import test_models

import pdb

# parameters
sequence = 4

image_dir = os.path.join(os.getcwd(), 'teaser_snapshots')
if not os.path.isdir(image_dir):
    os.mkdir(image_dir)

imageviz_transform = transforms.Compose([transforms.ToPILImage(),
    transforms.Resize((320, 896), interpolation=PIL.Image.NEAREST),
    transforms.ToTensor()])

# hack to get full (quarter) resolution image
orig_dataset = KITTIDerot('/mnt/tmp/data/kitti_derot/coarsesim_test', 
        sequences=[sequence], transform=None, input_scale=0, frame_offset=[1],
        pyramid_levels=[0], data_augmentation=False, fflip=-1, return_id=True)

hack_dataset = KITTIDerot('/mnt/tmp/data/kitti_derot/coarsesim_test', 
        sequences=[sequence], transform=None, input_scale=None, frame_offset=[1],
        pyramid_levels=[2, 3, 4], data_augmentation=False, fflip=-1, return_id=True)

frames = np.random.choice(orig_dataset.__len__(), 100)

for frame in frames:

    image_hr, _, _, _, _, _, _ = orig_dataset.__getitem__(frame)
    image_lr, _, _, _, _, _, _ = hack_dataset.__getitem__(frame)
        
    # shared images
    orig_name = os.path.join(image_dir, 'original_{:}_{:}.png'.format(frame, sequence))
    vutils.save_image(image_hr, orig_name, nrow=1)
    
    pixelated_name = os.path.join(image_dir, 'pixelated_{:}_{:}.png'.format(frame, sequence))
    vutils.save_image(imageviz_transform(image_lr), pixelated_name, nrow=1)

