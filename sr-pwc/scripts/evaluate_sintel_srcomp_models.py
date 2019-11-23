import sys
import os
import pprint
import math

import PIL

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

sys.path.append('..')

import data_utils
from data_utils import Sintel

import flow_utils

import test_models

import pdb

# parameters 
nvset = 100
model_index = sorted([0, 1, 2, 3, 4])
results_dir = '/home/sudipsin/sintel_srcomp_results'
if not os.path.isdir(results_dir):
    os.mkdir(results_dir)

# checkpoints
checkpoints = {}
checkpoints['pwc'] = os.path.join(os.getcwd(), 'states', 'sintel', 'pwc_190.pkl')
checkpoints['srresnet'] = os.path.join(os.getcwd(), 'states', 'sintel', 'srresnet_375.pkl')
checkpoints['srfnet'] = os.path.join(os.getcwd(), 'states', 'sintel', 'srpwc_190.pkl')

# models
model_list = [
        test_models.PWC_NO_SR(checkpoints['pwc']),
        test_models.PWC_ORACLE_SR(checkpoints['pwc']),
        test_models.PWC_ORACLE_SR(checkpoints['pwc']),
        test_models.SRFNet(checkpoints['srfnet']),
        test_models.SRFNet(checkpoints['srfnet'])
        ]

# transforms
input_transforms = [
        transforms.Compose([transforms.Resize((24, 48)), transforms.ToTensor()]), 
        transforms.Compose([transforms.Resize((48, 96)), transforms.ToTensor()]), 
        transforms.Compose([transforms.ToTensor()]),
        transforms.Compose([transforms.Resize((24, 48)), transforms.ToTensor()]), 
        transforms.Compose([transforms.Resize((24, 48)), transforms.ToTensor()])]
output_transforms = [
        transforms.Compose([flow_utils.ToFlow(), flow_utils.ResizeFlow((96, 192)), flow_utils.ToTensor()]),
        transforms.Compose([flow_utils.ToFlow(), flow_utils.ResizeFlow((96, 192)), flow_utils.ToTensor()]),
        None,
        transforms.Compose([flow_utils.ToFlow(), flow_utils.ResizeFlow((96, 192)), flow_utils.ToTensor()]),
        None]
output_flow_level = [0, 0, 0, 1, 0]
output_scaling = [4, 2, 1, 2, 1]

imageviz_transform = transforms.Compose([transforms.ToPILImage(),
    transforms.Resize((96, 192), interpolation=PIL.Image.NEAREST),
    transforms.ToTensor()])
flowviz_transform = transforms.Compose([flow_utils.ToFlow(),
    flow_utils.ToRGBImage(),
    transforms.ToTensor()])

for i in model_index:

    # model
    model = model_list[i]

    # for maintaining consistent training/validation splits
    torch.manual_seed(3607)
    
    # load new dataset
    dataset = Sintel('/mnt/tmp/data/sintel', data_augmentation=False,
            transform=input_transforms[i], target_transform=flow_utils.ToTensor(),
            pyramid_levels=[2, 3, 4], flow_scale=20.)
    _, validset = torch.utils.data.random_split(dataset, (dataset.__len__() - nvset, nvset))
    testloader = DataLoader(validset, batch_size=1, shuffle=False, num_workers=4)
    
    print('Evaluating model {:} ...'.format(model.model_id + str(i)))
    
    model_results = os.path.join(results_dir, model.model_id + str(i))
    if not os.path.isdir(model_results):
        os.mkdir(model_results)
    
    epe_data = []
    
    for image1, image2, flow_gt in testloader:
    
        with torch.no_grad():
    
            image1 = image1.cuda()
            image2 = image2.cuda()
            flow_gt = [flow.cuda() for flow in flow_gt]
    
            flow_hat = model(image1, image2)
            output_flow = flow_hat[output_flow_level[i]]
    
            if output_transforms[i] is not None:
                output_flow = torch.stack([output_transforms[i](flow).cuda() for flow in output_flow.detach().cpu()])

            epe = flow_utils.AEPE(5 * flow_gt[0], 5 * output_scaling[i] * output_flow).item()
    
        epe_data.append(epe)
    
    # write out summary data
    epe_data = np.array(epe_data)
    summary_file = os.path.join(model_results, 'epe_data.txt')
    np.savetxt(summary_file, epe_data, fmt='%.5f', delimiter=' ')
    
    # write out final statistics
    mean_epe = np.mean(epe_data)
    std_epe = np.std(epe_data)
    median_epe = np.median(epe_data)
    summary_statistcs = np.array([mean_epe, std_epe, median_epe])
    statistics_file = os.path.join(model_results, 'statistics.txt')
    np.savetxt(statistics_file, summary_statistcs, fmt='%.5f', delimiter=' ')

