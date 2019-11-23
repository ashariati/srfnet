import torch
import torch.nn as nn

import networks

import pdb

class PWC_NO_SR(nn.Module):

    def __init__(self, checkpoint):
        super(PWC_NO_SR, self).__init__()

        self.model_id = 'PWC_NO_SR'

        self.flow_model = networks.PWCNet().cuda()
        self.flow_model.load_state_dict(torch.load(checkpoint))

    def forward(self, x1, x2):
        y_hat = self.flow_model(x1, x2)
        return y_hat

class PWC_BICUBIC_SR(nn.Module):

    def __init__(self, checkpoint):
        super(PWC_BICUBIC_SR, self).__init__()

        self.model_id = 'PWC_BICUBIC_SR'

        self.flow_model = networks.PWCNet().cuda()
        self.flow_model.load_state_dict(torch.load(checkpoint))

    def forward(self, x1, x2):
        y_hat = self.flow_model(x1, x2)
        return y_hat

class PWC_SRResNet(nn.Module):

    def __init__(self, sr_checkpoint, flow_checkpoint):
        super(PWC_SRResNet, self).__init__()

        self.model_id = 'PWC_SRResNet'

        self.sr_model = networks.SRResNet().cuda()
        self.sr_model.load_state_dict(torch.load(sr_checkpoint))
        self.flow_model = networks.PWCNet().cuda()
        self.flow_model.load_state_dict(torch.load(flow_checkpoint))

    def forward(self, x1, x2):

        x1, _, _, _ = self.sr_model(x1)
        x2, _, _, _ = self.sr_model(x2)

        # transform to correct range of tensor image values
        x1 = 0.5 * (x1 + 1)
        x2 = 0.5 * (x2 + 1)

        y_hat = self.flow_model(x1, x2)
        return y_hat

class PWC_ORACLE_SR(nn.Module):

    def __init__(self, checkpoint):
        super(PWC_ORACLE_SR, self).__init__()

        self.model_id = 'PWC_ORACLE_SR'

        self.flow_model = networks.PWCNet().cuda()
        self.flow_model.load_state_dict(torch.load(checkpoint))

    def forward(self, x1, x2):
        y_hat = self.flow_model(x1, x2)
        return y_hat

class SRFNet(nn.Module):

    def __init__(self, checkpoint):
        super(SRFNet, self).__init__()

        self.model_id = 'SRFNet'

        self.joint_model = networks.SRPWCNet(
                networks.SRResNet().cuda(),
                networks.PWCNet().cuda(),
                freeze_pwc=False).cuda()

        self.joint_model.load_state_dict(torch.load(checkpoint))

    def forward(self, x1, x2):
        y_hat = self.joint_model(x1, x2)
        return y_hat

class SRFNet_K(nn.Module):

    def __init__(self, checkpoint):
        super(SRFNet_K, self).__init__()

        self.model_id = 'SRFNet_K'

        self.joint_model = networks.SRPWCNet(
                networks.SRResNet().cuda(),
                networks.PWCNet().cuda(),
                freeze_pwc=False).cuda()

        self.joint_model.load_state_dict(torch.load(checkpoint))

    def forward(self, x1, x2):
        y_hat = self.joint_model(x1, x2)
        return y_hat

class SRFNet_EK(nn.Module):

    def __init__(self, checkpoint):
        super(SRFNet_EK, self).__init__()

        self.model_id = 'SRFNet_EK'

        self.joint_model = networks.SRPWCNet(
                networks.SRResNet().cuda(),
                networks.PWCNet().cuda(),
                freeze_pwc=False).cuda()

        self.joint_model.load_state_dict(torch.load(checkpoint))

    def forward(self, x1, x2):
        y_hat = self.joint_model(x1, x2)
        return y_hat

