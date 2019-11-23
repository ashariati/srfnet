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
from data_utils import Sintel

import flow_utils
import test_models

import pdb

def put_text(img, text):
    np_img = flow_utils.to_flow(255 * img).copy()
    if np_img.shape[2] == 1:
        np_img = cv2.cvtColor(np_img, cv2.COLOR_GRAY2RGB)
    np_img = cv2.putText(np_img, text, (5, 15), cv2.FONT_HERSHEY_PLAIN, 1, [0, 0, 0], thickness=3)
    np_img = cv2.putText(np_img, text, (5, 15), cv2.FONT_HERSHEY_PLAIN, 1, [255, 255, 255])
    return F.to_tensor(np_img / 255)

# parameters
frame = 153
model_index = sorted([0, 1, 2, 3, 4])

image_dir = os.path.join(os.getcwd(), 'sintel_snapshots')
if not os.path.isdir(image_dir):
    os.mkdir(image_dir)

# checkpoints
checkpoints = {}
checkpoints['pwc'] = os.path.join(os.getcwd(), 'states', 'sintel', 'pwc_190.pkl')
checkpoints['srresnet'] = os.path.join(os.getcwd(), 'states', 'sintel', 'srresnet_375.pkl')
checkpoints['srfnet'] = os.path.join(os.getcwd(), 'states', 'sintel', 'srpwc_190.pkl')

# models
model_list = [
        test_models.PWC_NO_SR(checkpoints['pwc']),
        test_models.PWC_BICUBIC_SR(checkpoints['pwc']),
        test_models.PWC_SRResNet(checkpoints['srresnet'], checkpoints['pwc']),
        test_models.PWC_ORACLE_SR(checkpoints['pwc']),
        test_models.SRFNet(checkpoints['srfnet']),
        ]
model_names = [
        'PWC/None',
        'PWC/Bicubic',
        'PWC/SRResNet',
        'PWC/Oracle',
        'SRFNet-S',
        ]

# transforms
input_transforms = [
        transforms.Compose([transforms.Resize((24, 48)), transforms.ToTensor()]), 
        transforms.Compose([transforms.Resize((24, 48)),
            transforms.Resize((96, 192), interpolation=PIL.Image.BICUBIC), 
            transforms.ToTensor()]),
        transforms.Compose([transforms.Resize((24, 48)), transforms.ToTensor()]), 
        transforms.Compose([transforms.ToTensor()]),
        transforms.Compose([transforms.Resize((24, 48)), transforms.ToTensor()])]
output_transforms = [
        transforms.Compose([flow_utils.ToFlow(), flow_utils.ResizeFlow((96, 192)), flow_utils.ToTensor()]),
        None,
        None,
        None,
        None]
output_scaling = [4, 1, 1, 1, 1]

imageviz_transform = transforms.Compose([transforms.ToPILImage(),
    transforms.Resize((96, 192), interpolation=PIL.Image.NEAREST),
    transforms.ToTensor()])
flowviz_transform = transforms.Compose([flow_utils.ToFlow(),
    flow_utils.ToRGBImage(),
    transforms.ToTensor()])

# load new dataset
orig_dataset = Sintel('/mnt/tmp/data/sintel', data_augmentation=False,
        transform=transforms.Compose([transforms.Resize((96, 192)), transforms.ToTensor()]), 
        target_transform=flow_utils.ToTensor(), pyramid_levels=[2, 3, 4], flow_scale=20.)

for i in model_index:

    # model
    model = model_list[i]

    # load new dataset
    io_dataset = Sintel('/mnt/tmp/data/sintel', data_augmentation=False,
            transform=input_transforms[i], target_transform=flow_utils.ToTensor(), 
            pyramid_levels=[2, 3, 4], flow_scale=20.)

    image1, image2, flow_gt = io_dataset.__getitem__(frame)
    image1_orig, _, _ = orig_dataset.__getitem__(frame)

    # make batch
    image1 = image1[None, :, :, :]
    image2 = image2[None, :, :, :]

    with torch.no_grad():

        image1 = image1.cuda()
        image2 = image2.cuda()

        flow_hat = model(image1, image2)
        output_flow = flow_hat[0].detach()

        output_flow = output_scaling[i] * output_flow
        
        if output_transforms[i] is not None:
            output_flow = output_transforms[i](output_flow[0].cpu())
            output_flow = output_flow[None, :, :, :]

    flow_image_numpy = flow_utils.compute_flow_image(
            flow_utils.to_flow(output_flow[0].detach().cpu()))

    # save images
    flow_hat_name = os.path.join(image_dir, '%s_flow.png' % model.model_id)
    vutils.save_image(F.to_tensor(flow_image_numpy / 255), flow_hat_name, nrow=1)

# shared images
orig_name = os.path.join(image_dir, 'original.png')
vutils.save_image(image1_orig, orig_name, nrow=1)

pixelated_name = os.path.join(image_dir, 'pixelated.png')
vutils.save_image(imageviz_transform(image1[0].cpu()), pixelated_name, nrow=1)

flow_gt_name = os.path.join(image_dir, 'flow_gt.png')
vutils.save_image(flowviz_transform(flow_gt[0]), flow_gt_name, nrow=1)

