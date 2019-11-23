import sys
import os
import pprint
import math

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

import flow_utils

import networks
from networks import PWCNet

import pdb

# for maintaining consistent training/validation splits
torch.manual_seed(3607)

# training parameters
batch_size = 4
lr = 1e-5
gamma = 4e-4
alpha = [0.005, 0.01, 0.02]
start_epoch = 0
max_epochs = 500
log_interval = 10
save_interval = 5
things_end_epoch = 22

# visualizations
writer = SummaryWriter()
flowviz_transform = transforms.Compose([flow_utils.ToFlow(),
    flow_utils.ToRGBImage(),
    transforms.ToTensor()])

# data transforms
sample_transforms = transforms.Compose([
    transforms.ToTensor()])
target_transforms = transforms.Compose([
    flow_utils.ToTensor()])

# data 
nvset = 100
training_set = 'sintel'
dataset = Sintel('/mnt/tmp/data/sintel',
        transform=sample_transforms, target_transform=target_transforms,
        pyramid_levels=[2, 3, 4], flow_scale=20., crop_dim=(384, 768), hflip=0.5)
trainset, validset = torch.utils.data.random_split(dataset, (dataset.__len__() - nvset, nvset))

torch.manual_seed(3607 + start_epoch)
trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=4, shuffle=True)
validloader = DataLoader(validset, batch_size=10, num_workers=4)

# setup network
model = PWCNet().cuda()
if start_epoch > 0:
    checkpoint_file = os.path.join(os.getcwd(), 'states', training_set, 'pwc_%d.pkl' % start_epoch)
    model.load_state_dict(torch.load(checkpoint_file))
    print('Loading checkpoint')
else:
    checkpoint_file = os.path.join(os.getcwd(), 'states', 'flying_things', 'pwc_%d.pkl' % things_end_epoch)
    model.load_state_dict(torch.load(checkpoint_file))
    print('Loading initial network')


optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=2*gamma)
# scheduler = optim.lr_scheduler.StepLR(optimizer, 1e5, gamma=0.5)
# criterion = flow_utils.EPELoss()
criterion = flow_utils.RobustLoss(0.01, 0.4)

n_iter = start_epoch * math.ceil(float(trainset.__len__()) / batch_size)

model.train()
for epoch in range(start_epoch, max_epochs):

    for image1, image2, flow_gt in trainloader:

        image1 = image1.cuda()
        image2 = image2.cuda()
        flow_gt = [flow.cuda() for flow in flow_gt]

        flow_hat = model(image1, image2)
        loss = alpha[0] * criterion(flow_hat[0], flow_gt[0]) + \
            alpha[1] * criterion(flow_hat[1], flow_gt[1]) + \
            alpha[2] * criterion(flow_hat[2], flow_gt[2])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n_iter += 1

        # scheduler.step()

        if n_iter % log_interval == 0:
            print('epoch [{}/{}], iteration [{}], loss: {:.5f}'.format(epoch + 1,
                max_epochs, n_iter, loss.item()))

            # visualizations
            writer.add_scalar('data/loss', loss.item(), n_iter)

            n_samples = flow_gt[0].shape[0]
            flow_hat_image = torch.stack([flowviz_transform(flow_hat[0].detach().cpu()[i, :, :, :]) for i in range(n_samples)])
            flow_gt_image = torch.stack([flowviz_transform(flow_gt[0].detach().cpu()[i, :, :, :]) for i in range(n_samples)])
            
            flow_image = torch.cat((image1.cpu(), flow_gt_image, flow_hat_image))
            
            progress_images = vutils.make_grid(flow_image, nrow=batch_size)
            writer.add_image('flows', progress_images, n_iter)


    if (epoch + 1) % save_interval == 0:
        checkpoint_file = os.path.join(os.getcwd(), 'states', training_set,'pwc_%d.pkl' % (epoch + 1))
        torch.save(model.state_dict(), checkpoint_file)

    verror = 0
    viters = 0
    for image1, image2, flow_gt in validloader:

        image1 = image1.cuda()
        image2 = image2.cuda()
        flow_gt = [flow.cuda() for flow in flow_gt]

        flow_hat = model(image1, image2)

        verror += flow_utils.AEPE(5 * flow_gt[0], 5 * flow_hat[0]).item()
        viters += 1

    writer.add_scalar('data/aepe', verror / viters, n_iter)

        
writer.close()

