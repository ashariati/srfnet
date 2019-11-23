import sys
import os
import pprint

import numpy as np

import PIL

import torch
from torch.utils.data import DataLoader
import torchvision.utils as vutils

import torchvision.transforms as transforms

sys.path.append('../..')

from data_utils import KITTIDerot
from networks import SRPWCNet
from networks import SRResNet
from networks import PWCNet

import flow_utils
from flow_utils import ToTensor, ToPILImage, ToFlow

import pdb

torch.manual_seed(3607)

flowviz_transform = transforms.Compose([flow_utils.ToFlow(),
    flow_utils.ToRGBImage(),
    transforms.ToTensor()])

imageviz_transform = transforms.Compose([transforms.ToPILImage(),
    transforms.Resize((80, 224), interpolation=PIL.Image.NEAREST),
    transforms.ToTensor()])

batch_size = 10

dataset = KITTIDerot('/mnt/tmp/data/kitti_derot/coarsesim', sequences=[2], pyramid_levels=[2, 3, 4], fflip=-1)
loader = DataLoader(dataset, batch_size=batch_size, num_workers=2, shuffle=True)

last_epoch = 50
# checkpoint_file = os.path.join(os.getcwd(), '..', 'states', 'sintel', 'srpwc_%d.pkl' % 190)
checkpoint_file = os.path.join(os.getcwd(), '..', 'states', 'kitti_derot', 'srpwc_%d.pkl' % last_epoch)
model = SRPWCNet(SRResNet().cuda(), PWCNet().cuda(), freeze_pwc=False).cuda()
model.load_state_dict(torch.load(checkpoint_file))
criterion = flow_utils.EpipolarLoss(0.5, 10 * (np.pi / 180))

to_pil = transforms.ToPILImage()

image1, image2, flow, K, t = dataset.__getitem__(3 * 559)

for image1, image2, flow_gt, K, t in loader:

        # to_pil(image1[0]).show()
        # to_pil(image2[0]).show()

        # to_pil(image1[1]).show()
        # to_pil(image2[1]).show()

        image1 = image1.cuda()
        image2 = image2.cuda()
        flow_gt = [flow.cuda() for flow in flow_gt]

        flow_hat = model(image1, image2)

        # loss = criterion(flow_hat[0], flow_gt[0])
        
        epe, pcent, theta, in_mask, error_image = flow_utils.trans_error(flow_hat[0].detach(), K, t, debug=True)

        n_samples = flow_gt[0].shape[0]
        image1_resized = torch.stack([imageviz_transform(image1.cpu()[i, :, :, :]) for i in range(n_samples)])
        image2_resized = torch.stack([imageviz_transform(image2.cpu()[i, :, :, :]) for i in range(n_samples)])
        flow_hat_image = torch.stack([flowviz_transform(flow_hat[0].detach().cpu()[i, :, :, :]) for i in range(n_samples)])

        debug_tensor1 = torch.cat((flow_hat_image, image1_resized, image2_resized))
        debug_tensor2 = torch.cat((255 * in_mask.float(), torch.clamp((1. / 10) * (error_image), 0, 255)))
        # debug_image = vutils.make_grid(debug_tensor, nrow=batch_size)
        vutils.save_image(debug_tensor1, './debug_images/test1.png', nrow=batch_size)
        vutils.save_image(debug_tensor2, './debug_images/test2.png', nrow=batch_size)
        
        pdb.set_trace()

# pil_transform = ToPILImage()
# pil_transform(flow_hat[0].detach().cpu()[0, :, :, :]).show()
# pil_transform(flow_gt[0]).show()
# image1.show()
# image2.show()



pdb.set_trace()
