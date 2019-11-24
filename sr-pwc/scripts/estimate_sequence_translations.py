import sys
import os

import PIL

import numpy as np

import torch
from torch.utils.data import DataLoader

import torchvision.transforms as transforms

sys.path.append('..')

import data_utils
from data_utils import KITTIDerot

import flow_utils

import test_models

# parameters
frame_offset = [1, 2, 3]
window_size = len(frame_offset)
sequence = 7
results_dir = '/home/armon/kitti_derot_results'
if not os.path.isdir(results_dir):
    os.mkdir(results_dir)

checkpoint_file = os.path.join(os.getcwd(), 'states', 'kitti_derot', 'srpwc_22_s0123.pkl')
model = test_models.SRFNet_EK(checkpoint_file)

# transforms
imageviz_transform = transforms.Compose([transforms.ToPILImage(),
    transforms.Resize((80, 224), interpolation=PIL.Image.NEAREST),
    transforms.ToTensor()])
flowviz_transform = transforms.Compose([flow_utils.ToFlow(),
    flow_utils.ToRGBImage(),
    transforms.ToTensor()])

print('Loading sequence {:d} ...'.format(sequence))

# setup sequence directory
sequence_results = os.path.join(results_dir, "%02d" % sequence)
if not os.path.isdir(sequence_results):
    os.mkdir(sequence_results)

# load new dataset
dataset = KITTIDerot('/home/armon/Research/Data/srfnet/kitti/coarsesim', sequences=[sequence],
        transform=None, input_scale=None, frame_offset=frame_offset,
        pyramid_levels=[2, 3, 4], data_augmentation=False, fflip=-1, return_id=True)
testloader = DataLoader(dataset, batch_size=window_size, shuffle=False, num_workers=4)

nsamples = dataset.__len__()
npairs = int(nsamples / window_size)

translations = []

for image1, image2, flow_gt, K, t, subseq_id, pair_id in testloader:

    print('\rProcessing subsequence {:d} / {:d}'.format(int(subseq_id[0]), npairs), end="")

    with torch.no_grad():

        image1 = image1.cuda()
        image2 = image2.cuda()
        flow_gt = [flow.cuda() for flow in flow_gt]

        flow_hat = model(image1, image2)
        output_flow = flow_hat[0]

        epe, pcent, theta, t_hat, in_mask, _ = flow_utils.trans_error(output_flow.detach(), K, t, debug=True)

    translations.extend(t_hat)

# write translations
translations = np.array(translations)
translations_file = os.path.join(sequence_results, 'translations.txt')
np.savetxt(translations_file, translations, fmt='%.5f', delimiter=' ')


