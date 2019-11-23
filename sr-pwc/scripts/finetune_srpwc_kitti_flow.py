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
from data_utils import FlyingChairs
from data_utils import FlyingThings
from data_utils import KITTIFlow

import flow_utils

import networks
from networks import SRPWCNet
from networks import PWCNet
from networks import SRResNet

import pdb

# for maintaining consistent training/validation splits
torch.manual_seed(3607)

# training parameters
batch_size = 4
lr = 1e-5
alpha = [0.005, 0.01, 0.02]
start_epoch = 0
max_epochs = 500
log_interval = 10
save_interval = 5
sintel_end_epoch = 190

# visualizations
writer = SummaryWriter()
flowviz_transform = transforms.Compose([flow_utils.ToFlow(),
    flow_utils.ToRGBImage(),
    transforms.ToTensor()])

imageviz_transform = transforms.Compose([transforms.ToPILImage(),
    transforms.Resize((80, 224), interpolation=PIL.Image.NEAREST),
    transforms.ToTensor()])

# data transforms
sample_transforms = transforms.Compose([
    transforms.Resize((20, 56)),
    transforms.ToTensor()])

# data 
nvset = 20
training_set = 'kitti_flow'
dataset = KITTIFlow('/mnt/tmp/data/kitti_flow',
        transform=sample_transforms, pyramid_levels=[2, 3, 4], flow_scale=20., crop_dim=(320, 896))
trainset, validset = torch.utils.data.random_split(dataset, (dataset.__len__() - nvset, nvset))

torch.manual_seed(3607 + start_epoch)
trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=4, shuffle=True)
validloader = DataLoader(validset, batch_size=1, num_workers=4)

# setup network
model = SRPWCNet(SRResNet().cuda(), PWCNet().cuda(), freeze_pwc=False).cuda()
if start_epoch > 0:
    checkpoint_file = os.path.join(os.getcwd(), 'states', training_set, 'srpwc_%d.pkl' % start_epoch)
    model.load_state_dict(torch.load(checkpoint_file))
    print('Loading checkpoint')
else:
    checkpoint_file = os.path.join(os.getcwd(), 'states', 'sintel', 'srpwc_%d.pkl' % sintel_end_epoch)
    model.load_state_dict(torch.load(checkpoint_file))
    print('Loading Sintel model')

optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, 1e5, gamma=0.5)
criterion = flow_utils.MaskedRobustLoss(0.01, 0.2)

n_iter = start_epoch * math.ceil(float(trainset.__len__()) / batch_size)

model.train()
for epoch in range(start_epoch, max_epochs):

    for image1, image2, flow_gt, masks in trainloader:

        image1 = image1.cuda()
        image2 = image2.cuda()
        flow_gt = [flow.cuda() for flow in flow_gt]
        masks = [mask.cuda() for mask in masks]

        flow_hat = model(image1, image2)
        loss = alpha[0] * criterion(flow_hat[0], flow_gt[0], masks[0]) + \
            alpha[1] * criterion(flow_hat[1], flow_gt[1], masks[1]) + \
            alpha[2] * criterion(flow_hat[2], flow_gt[2], masks[2])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n_iter += 1

        scheduler.step()

        if n_iter % log_interval == 0:
            print('epoch [{}/{}], iteration [{}], loss: {:.5f}'.format(epoch + 1,
                max_epochs, n_iter, loss.item()))

            # visualizations
            writer.add_scalar('data/loss', loss.item(), n_iter)

            n_samples = flow_gt[0].shape[0]
            flow_hat_image = torch.stack([flowviz_transform(flow_hat[0].detach().cpu()[i, :, :, :]) for i in range(n_samples)])
            flow_gt_image = torch.stack([flowviz_transform(flow_gt[0].detach().cpu()[i, :, :, :]) for i in range(n_samples)])
            image1_resized = torch.stack([imageviz_transform(image1.cpu()[i, :, :, :]) for i in range(n_samples)])
            
            flow_image = torch.cat((image1_resized, flow_gt_image, flow_hat_image))
            
            progress_images = vutils.make_grid(flow_image, nrow=batch_size)
            writer.add_image('flows', progress_images, n_iter)


    if (epoch + 1) % save_interval == 0:
        checkpoint_file = os.path.join(os.getcwd(), 'states', training_set, 'srpwc_%d.pkl' % (epoch + 1))
        torch.save(model.state_dict(), checkpoint_file)

    verror = 0
    viters = 0
    for image1, image2, flow_gt, masks in validloader:

        image1 = image1.cuda()
        image2 = image2.cuda()
        flow_gt = [flow.cuda() for flow in flow_gt]
        masks = [mask.cuda() for mask in masks]

        flow_hat = model(image1, image2)

        verror += flow_utils.MaskedAEPE(5 * flow_gt[0], 5 * flow_hat[0], masks[0]).item()
        viters += 1

    writer.add_scalar('data/aepe', verror / viters, n_iter)

        
writer.close()

