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

def put_text(img, text):
    np_img = flow_utils.to_flow(255 * img).copy()
    if np_img.shape[2] == 1:
        np_img = cv2.cvtColor(np_img, cv2.COLOR_GRAY2RGB)
    np_img = cv2.putText(np_img, text, (5, 15), cv2.FONT_HERSHEY_PLAIN, 1, [0, 0, 0], thickness=3)
    np_img = cv2.putText(np_img, text, (5, 15), cv2.FONT_HERSHEY_PLAIN, 1, [255, 255, 255])
    return F.to_tensor(np_img / 255)

# parameters
sequence = 7
frame = 52
# sequence = 7
# frame = 56
model_index = sorted([0, 1, 2, 3, 4, 5, 6])

image_dir = os.path.join(os.getcwd(), 'kitti_snapshots_{:}_{:}'.format(frame, sequence))
if not os.path.isdir(image_dir):
    os.mkdir(image_dir)

# checkpoints
checkpoints = {}
checkpoints['pwc'] = os.path.join(os.getcwd(), 'states', 'kitti_flow', 'pwc_445.pkl')
checkpoints['srresnet'] = os.path.join(os.getcwd(), 'states', 'kitti_flow', 'srresnet_410.pkl')
checkpoints['srfnet'] = os.path.join(os.getcwd(), 'states', 'sintel', 'srpwc_190.pkl')
checkpoints['srfnet-k'] = os.path.join(os.getcwd(), 'states', 'kitti_flow', 'srpwc_500.pkl')
checkpoints['srfnet-ek'] = os.path.join(os.getcwd(), 'states', 'kitti_derot', 'srpwc_22_s0123.pkl')

# models
model_list = [
        test_models.PWC_NO_SR(checkpoints['pwc']),
        test_models.PWC_BICUBIC_SR(checkpoints['pwc']),
        test_models.PWC_SRResNet(checkpoints['srresnet'], checkpoints['pwc']),
        test_models.PWC_ORACLE_SR(checkpoints['pwc']),
        test_models.SRFNet(checkpoints['srfnet']),
        test_models.SRFNet_K(checkpoints['srfnet-k']),
        test_models.SRFNet_EK(checkpoints['srfnet-ek'])
        ]
model_names = [
        'PWC/None',
        'PWC/Bicubic',
        'PWC/SRResNet',
        'PWC/Oracle',
        'SRFNet-S',
        'SRFNet-K',
        'SRFNet-EK',
        ]

# transforms
input_scales = [None, None, None, 2, None, None, None]
input_transforms = [
        None,
        transforms.Resize((80, 224), interpolation=PIL.Image.BICUBIC),
        None,
        None,
        None,
        None,
        None]
output_transforms = [
        transforms.Compose([flow_utils.ToFlow(), flow_utils.ResizeFlow((80, 224)), flow_utils.ToTensor()]),
        None,
        None,
        None,
        None,
        None,
        None]
output_scaling = [4, 1, 1, 1, 1, 1, 1]

imageviz_transform = transforms.Compose([transforms.ToPILImage(),
    transforms.Resize((80, 224), interpolation=PIL.Image.NEAREST),
    transforms.ToTensor()])
flowviz_transform = transforms.Compose([flow_utils.ToFlow(),
    flow_utils.ToRGBImage(),
    transforms.ToTensor()])

# hack to get full (quarter) resolution image
orig_dataset = KITTIDerot('/mnt/tmp/data/kitti_derot/coarsesim_test', 
        sequences=[sequence], transform=None, input_scale=2, frame_offset=[1],
        pyramid_levels=[2], data_augmentation=False, fflip=-1, return_id=True)

for i in model_index:

    # model
    model = model_list[i]

    # load new dataset
    io_dataset = KITTIDerot('/mnt/tmp/data/kitti_derot/coarsesim_test', sequences=[sequence], 
            transform=input_transforms[i], input_scale=input_scales[i], frame_offset=[1],
            pyramid_levels=[2, 3, 4], data_augmentation=False, fflip=-1, return_id=True)

    image1, image2, flow_gt, K, t, subseq_id, pair_id = io_dataset.__getitem__(frame)
    image1_orig, _, _, _, _, _, _ = orig_dataset.__getitem__(frame)

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

        _, _, _, t_hat, in_mask, _ = flow_utils.trans_error(output_flow, [K], [t], debug=True)

    # custom visualization
    flow_image_numpy = flow_utils.compute_flow_image(
            flow_utils.to_flow(output_flow[0].detach().cpu()))
    in_mask_numpy = flow_utils.to_flow(in_mask[0])
    masked_flow_numpy = np.multiply(flow_image_numpy, in_mask_numpy) / 255
    lowres_numpy = flow_utils.to_flow(imageviz_transform(image1[0].cpu()))
    blended_numpy = 0.5 * lowres_numpy + 0.5 * masked_flow_numpy

    # draw epipole
    e = np.dot(K, t)
    e = e / e[2]
    e_hat = np.dot(K, t_hat[0])
    e_hat = e_hat / e_hat[2]
    flow_image_numpy = cv2.circle(flow_image_numpy, (int(e[0]), int(e[1])), 3, [255, 0, 0], thickness=-1)
    flow_image_numpy = cv2.circle(flow_image_numpy, (int(e_hat[0]), int(e_hat[1])), 3, [0, 0, 255], thickness=-1)

    # save images
    blended_flow_name = os.path.join(image_dir, '%s_blended.png' % model.model_id)
    vutils.save_image(F.to_tensor(blended_numpy), blended_flow_name, nrow=1)

    flow_hat_name = os.path.join(image_dir, '%s_flow.png' % model.model_id)
    vutils.save_image(F.to_tensor(flow_image_numpy / 255), flow_hat_name, nrow=1)

    mask_name = os.path.join(image_dir, '%s_mask.png' % model.model_id)
    vutils.save_image(in_mask[0], mask_name, nrow=1)
    
# shared images
orig_name = os.path.join(image_dir, 'original.png')
vutils.save_image(image1_orig, orig_name, nrow=1)

pixelated_name = os.path.join(image_dir, 'pixelated.png')
vutils.save_image(imageviz_transform(image1[0].cpu()), pixelated_name, nrow=1)

flow_gt_name = os.path.join(image_dir, 'flow_gt.png')
vutils.save_image(flowviz_transform(flow_gt[0]), flow_gt_name, nrow=1)

