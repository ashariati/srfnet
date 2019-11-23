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
from data_utils import KITTIDerot

import flow_utils

import test_models

import pdb

# parameters 
frame_offset = [1]
window_size = len(frame_offset)
sequences = [4, 5, 6, 7, 8, 9, 10]
model_index = sorted([0, 1, 2, 3, 4, 5, 6])
results_dir = '/home/t-arsha/kitti_derot_results'
if not os.path.isdir(results_dir):
    os.mkdir(results_dir)

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
imageviz_transform = transforms.Compose([transforms.ToPILImage(),
    transforms.Resize((80, 224), interpolation=PIL.Image.NEAREST),
    transforms.ToTensor()])
flowviz_transform = transforms.Compose([flow_utils.ToFlow(),
    flow_utils.ToRGBImage(),
    transforms.ToTensor()])

for sequence in sequences:

    print('Loading sequence {:d} ...'.format(sequence))

    # setup sequence directory
    sequence_results = os.path.join(results_dir, "%02d" % sequence)
    if not os.path.isdir(sequence_results):
        os.mkdir(sequence_results)

    for offset in frame_offset:

        # setup offset directory
        offset_results = os.path.join(sequence_results, "offset_%d" % offset)
        if not os.path.isdir(offset_results):
            os.mkdir(offset_results)

        for i in model_index:

            # model
            model = model_list[i]

            # load new dataset
            dataset = KITTIDerot('/mnt/tmp/data/kitti_derot/coarsesim_test', sequences=[sequence], 
                    transform=input_transforms[i], input_scale=input_scales[i], frame_offset=[offset],
                    pyramid_levels=[2, 3, 4], data_augmentation=False, fflip=-1, return_id=True)
            testloader = DataLoader(dataset, batch_size=window_size, shuffle=False, num_workers=4)

            nsamples = dataset.__len__()
            npairs = int(nsamples / window_size)

            print('Evaluating model {:} ...'.format(model.model_id))

            model_results = os.path.join(offset_results, model.model_id)
            if not os.path.isdir(model_results):
                os.mkdir(model_results)

            imagedir = os.path.join(model_results, 'images')
            if not os.path.isdir(imagedir):
                os.mkdir(imagedir)

            maskdir = os.path.join(model_results, 'inlier_masks')
            if not os.path.isdir(maskdir):
                os.mkdir(maskdir)

            flowdir = os.path.join(model_results, 'flows')
            if not os.path.isdir(flowdir):
                os.mkdir(flowdir)

            imflowdir = os.path.join(model_results, 'flow_images')
            if not os.path.isdir(imflowdir):
                os.mkdir(imflowdir)

            summary_data = []
            translations = []

            for image1, image2, flow_gt, K, t, subseq_id, pair_id in testloader:

                print('\rProcessing subsequence {:d} / {:d}'.format(int(subseq_id[0]), npairs), end="")

                with torch.no_grad():

                    image1 = image1.cuda()
                    image2 = image2.cuda()
                    flow_gt = [flow.cuda() for flow in flow_gt]

                    flow_hat = model(image1, image2)
                    output_flow = flow_hat[0]

                    if output_transforms[i] is not None:
                        output_flow = torch.stack([output_transforms[i](flow) for flow in output_flow.detach().cpu()])

                    epe, pcent, theta, t_hat, in_mask, _ = flow_utils.trans_error(output_flow.detach(), K, t, debug=True)

                summary_data.append(np.concatenate((epe, pcent, theta)))
                translations.extend(t_hat)

                # save images
                # for j in range(window_size):
                #     # imname = os.path.join(imagedir, '{:}_{:}.png'.format(subseq_id[j], pair_id[j]))
                #     # vutils.save_image(imageviz_transform(image1[j].cpu()), imname, nrow=1)
                #     # imflowname = os.path.join(imflowdir, '{:}_{:}.png'.format(subseq_id[j], pair_id[j]))
                #     # vutils.save_image(flowviz_transform(output_flow[j].detach().cpu()), imflowname, nrow=1)
                #     maskname = os.path.join(maskdir, '{:}_{:}.png'.format(subseq_id[j], pair_id[j]))
                #     vutils.save_image(in_mask[j], maskname, nrow=1)
                #     flowname = os.path.join(flowdir, '{:}_{:}.flo'.format(subseq_id[j], pair_id[j]))
                #     data_utils.write_flow(flowname, flow_utils.to_flow(output_flow[j].detach().cpu()))

            print('')

            # write out summary data
            summary_data = np.array(summary_data)
            summary_file = os.path.join(model_results, 'summary_data.txt')
            np.savetxt(summary_file, summary_data, fmt='%.5f', delimiter=' ')

            # write out final statistics
            epes = summary_data[:, :window_size]
            pcents = summary_data[:, window_size:2*window_size]
            thetas = summary_data[:, 2*window_size:]
            mean_epe = np.mean(epes)
            mean_pcent = np.mean(pcents)
            mean_theta = np.mean(thetas)
            std_epe = np.std(epes)
            std_pcent = np.std(pcents)
            std_theta = np.std(thetas)
            median_epe = np.median(epes)
            median_pcent = np.median(pcents)
            median_theta = np.median(thetas)
            summary_statistcs = np.array([[mean_epe, mean_pcent, mean_theta],
                    [std_epe, std_pcent, std_theta],
                    [median_epe, median_pcent, median_theta]])
            statistics_file = os.path.join(model_results, 'statistics.txt')
            np.savetxt(statistics_file, summary_statistcs, fmt='%.5f', delimiter=' ')

            # write translations
            translations = np.array(translations)
            translations_file = os.path.join(model_results, 'translations.txt')
            np.savetxt(translations_file, translations, fmt='%.5f', delimiter=' ')


