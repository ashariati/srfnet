import sys
import os
import pprint

import numpy as np

import torch

import torchvision.transforms as transforms

sys.path.append('../..')

from data_utils import Sintel
from data_utils import FlyingChairs
from data_utils import FlyingChairsSR
from data_utils import FlyingThings
from data_utils import KITTIFlow
from data_utils import KITTIDerot

from flow_utils import ToTensor, ToPILImage

import pdb

# dataset = Sintel('/data1/data/sintel', split='training', hflip=0.5)
# dataset = FlyingChairs('/data1/data/flying_chairs/FlyingChairs_release/data')
# dataset = FlyingChairsSR('/home/armon/Projects/CVIO/data/flying_chairs/FlyingChairs_release/data')
# dataset = FlyingThings('/data1/data/flying_things')
# dataset = KITTIFlow('/mnt/tmp/data/kitti_flow', pyramid_levels=[0, 2, 3, 4])
dataset = KITTIDerot('/mnt/tmp/data/kitti_derot/coarsesim_test', sequences=[4], pyramid_levels=[2, 3, 4], fflip=-1, frame_offset=[2])

image1, image2, flow, mask = dataset.__getitem__(3 * 559)

image1.show()
image2.show()

pil_transform = ToPILImage()
tensor_transform = ToTensor()
# pil_transform(tensor_transform(flow[0])).show()
# pil_transform(tensor_transform(flow[1])).show()
# pil_transform(tensor_transform(flow[-1])).show()
pil_transform(flow[0]).show()
pil_transform(flow[1]).show()
pil_transform(flow[-1]).show()

pdb.set_trace()
