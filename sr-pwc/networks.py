import sys
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from correlation_package.correlation import Correlation 
from layers import ResBlock2
from layers import SubPixConv
from network_utils import init_weights

import pdb

def freeze_layer(layer):
    for param in layer.parameters():
        param.requires_grad = False
    return layer

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):   
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, 
                        padding=padding, dilation=dilation, bias=True),
            nn.LeakyReLU(0.1))

def predict_flow(in_planes):
    return nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=True)

def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class DenseNet(nn.Module):

    def __init__(self, in_channels, out_channels, 
            kernel_size=3, stride=1, padding=1):
        super(DenseNet, self).__init__()

        layers = []
        n_in = np.cumsum([0] + out_channels)
        for i, oc in enumerate(out_channels):
            layers.append(conv(in_channels + n_in[i], oc))

        self.module_list = nn.ModuleList(layers)

        self.apply(init_weights)

    def forward(self, x):

        for layer in self.module_list:
            x = torch.cat((layer(x), x), 1)

        return x

class SRResNet(nn.Module):

    def __init__(self):
        super(SRResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4)
        self.relu1 = nn.PReLU()

        self.resblocks = nn.Sequential(*[ResBlock2(64, 64) for _ in range(16)])

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.PReLU()

        self.espc1 = nn.Sequential(
                SubPixConv(64, out_channels=128, upscale_factor=2),
                nn.PReLU())
        self.espc2 = nn.Sequential(
                SubPixConv(32, out_channels=64, upscale_factor=2),
                nn.PReLU())

        self.conv4 = nn.Conv2d(16, 3, kernel_size=9, stride=1, padding=4)

        self.apply(init_weights)

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu1(x)
        xf = x

        x = self.resblocks(x)
        x = self.conv2(x)

        x = x + xf

        x = self.conv3(x)
        x = self.relu3(x)

        c1 = x
        c2 = self.espc1(c1)
        c3 = self.espc2(c2)

        x = self.conv4(c3)

        return x, c1, c2, c3

class FeaturePyramidExtractor(nn.Module):

    def __init__(self):
        super(FeaturePyramidExtractor, self).__init__()

        self.module_list = nn.ModuleList([
            nn.Sequential(conv(3,  16, stride=1), conv(16, 16)),
            nn.Sequential(conv(16, 32, stride=2), conv(32, 32)),
            nn.Sequential(conv(32, 64, stride=2), conv(64, 64))
            ])

        self.apply(init_weights)

    def forward(self, x):

        c = []
        for i, layer in enumerate(self.module_list):
            x = layer(x)
            c.append(x)

        return c[::-1]

class OpticalFlowEstimator(nn.Module):

    def __init__(self, in_channels):
        super(OpticalFlowEstimator, self).__init__()

        out_channels = [128,128,96,64,32]
        dense_out_channel = np.sum(out_channels)
        self.dense_net = DenseNet(in_channels, out_channels)
        self.predict = predict_flow(in_channels + dense_out_channel)

        self.apply(init_weights)

    def forward(self, x):

        c = self.dense_net(x)
        x = self.predict(c)

        return x, c

class ContextNet(nn.Module):

    def __init__(self, in_channels):
        super(ContextNet, self).__init__()

        self.conv_block = nn.Sequential(
                conv(in_channels, 128, kernel_size=3, stride=1, padding=1,  dilation=1),
                conv(128, 128, kernel_size=3, stride=1, padding=2, dilation=2),
                conv(128, 128, kernel_size=3, stride=1, padding=4, dilation=4),
                conv(128, 96, kernel_size=3, stride=1, padding=8, dilation=8),
                conv(96, 64, kernel_size=3, stride=1, padding=16, dilation=16),
                conv(64, 32, kernel_size=3, stride=1, padding=1, dilation=1),
                )

        self.predict = predict_flow(32)

        self.apply(init_weights)

    def forward(self, x):

        x = self.conv_block(x)
        x = self.predict(x)

        return x

class WarpingLayer(nn.Module):

    def __init__(self):
        super(WarpingLayer, self).__init__()

    def forward(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """
        B, C, H, W = x.size()
        # mesh grid 
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = Variable(grid) + flo

        # scale grid to [-1,1] 
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

        vgrid = vgrid.permute(0,2,3,1)        
        output = nn.functional.grid_sample(x, vgrid)
        mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid)
        
        mask[mask<0.9999] = 0
        mask[mask>0] = 1
        
        return output*mask

class PWCNet(nn.Module):

    def __init__(self, md=4, level_channels=[16, 32, 64]):
        super(PWCNet, self).__init__()

        level_channels = sorted(level_channels)
        self.n_levels = len(level_channels)

        self.pyramid_extraction = FeaturePyramidExtractor()

        self.warp = WarpingLayer()

        self.corr = Correlation(pad_size=md, kernel_size=1, max_displacement=md, 
                stride1=1, stride2=1, corr_multiply=1)
        self.relu = nn.LeakyReLU(0.1)

        self.flow_scales = [1.25, 2.5, 5]

        nd = (2*md+1)**2
        self.flow_estimators = nn.ModuleList([
                OpticalFlowEstimator(nd + level_channels[2]),
                OpticalFlowEstimator(nd + level_channels[1] + 2),
                OpticalFlowEstimator(nd + level_channels[0] + 2)])

        self.upsampling_layers = nn.ModuleList([
            Identity(),
            SubPixConv(2, out_channels=8, upscale_factor=2),
            SubPixConv(2, out_channels=8, upscale_factor=2)])

        self.context_network = ContextNet((nd + level_channels[0] + 448 + 2) + 2)

        self.apply(init_weights)


    def forward(self, im1, im2):

        # extract features
        c1 = self.pyramid_extraction(im1)
        c2 = self.pyramid_extraction(im2)

        # initial flow
        flowshape = list(c1[0].size())
        flowshape[1] = 2
        flow = torch.zeros(flowshape).to(im1.device)

        flows = []
        for l in range(self.n_levels):

            up_flow = self.upsampling_layers[l](flow)

            c2_warp = self.warp(c2[l], up_flow * self.flow_scales[l])

            corr = self.corr(c1[l], c2_warp)
            corr = self.relu(corr)

            if l == 0:
                xf = torch.cat([corr, c1[l]], 1)
            else:
                xf = torch.cat([corr, c1[l], up_flow], 1)

            flow, flow_features = self.flow_estimators[l](xf)

            flows.append(flow)

        refined_residual_flow = self.context_network(torch.cat([flow, flow_features], 1))
        flows[-1] = flows[-1] + refined_residual_flow

        if self.training:
            return flows[::-1]
        else:
            return flows[-1]

class SRPWCNet(nn.Module):

    def __init__(self, srresnet, pwcnet, freeze_pwc=False):
        super(SRPWCNet, self).__init__()

        self.srresnet = srresnet

        if freeze_pwc:
            pwcnet = freeze_layer(pwcnet)

        self.n_levels = pwcnet.n_levels
        self.flow_scales = pwcnet.flow_scales

        self.warp = pwcnet._modules['warp']
        self.corr = pwcnet._modules['corr']
        self.relu = pwcnet._modules['relu']
        self.flow_estimators = pwcnet._modules['flow_estimators']
        self.upsampling_layers = pwcnet._modules['upsampling_layers']
        self.context_network = pwcnet._modules['context_network']

    def forward(self, im1, im2):

        # extract features
        c1 = self.srresnet(im1)[1:]
        c2 = self.srresnet(im2)[1:]

        # initial flow
        flowshape = list(c1[0].size())
        flowshape[1] = 2
        flow = torch.zeros(flowshape).to(im1.device)

        flows = []
        for l in range(self.n_levels):

            up_flow = self.upsampling_layers[l](flow)

            c2_warp = self.warp(c2[l], up_flow * self.flow_scales[l])

            corr = self.corr(c1[l], c2_warp)
            corr = self.relu(corr)

            if l == 0:
                xf = torch.cat([corr, c1[l]], 1)
            else:
                xf = torch.cat([corr, c1[l], up_flow], 1)

            flow, flow_features = self.flow_estimators[l](xf)

            flows.append(flow)

        refined_residual_flow = self.context_network(torch.cat([flow, flow_features], 1))
        flows[-1] = flows[-1] + refined_residual_flow

        if self.training:
            return flows[::-1]
        else:
            return flows[-1]

