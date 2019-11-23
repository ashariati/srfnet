import os
import itertools
import glob
import re

import torch 
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

import PIL
from PIL import Image

import cv2

import numpy as np

import matplotlib.pyplot as plt

import flow_utils

import pdb

SINTEL_DIM = (1024, 436)
CHAIRS_DIM = (512, 384)
THINGS_DIM = (960, 540)
KITTI_DIM = (1241, 376)
KITTI_COARSE_DIM = (77, 23)

class MaskToTensor(object):
    def __call__(self, x):
        return to_tensor(x)

def to_tensor(mask):
    mask = mask[:, :, None]
    return torch.from_numpy(mask.transpose((2, 0, 1))).byte()

def kitti_invalid_mask(image):
    image = np.array(image)
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask[:, :, 2] = 255 * np.ones(image.shape[:2], dtype=np.uint8)
    return np.all(mask == image, axis=2).astype(np.uint8)

class InpaintNaNs(object):

    def __call__(self, x):

        if isinstance(x, Image.Image):

            x = np.array(x)
            nanmask = np.isnan(x)

            if not np.any(nanmask):
                x = Image.fromarray(x)
                return x

            x = cv2.inpaint(x, nanmask.astype(np.uint8), 2, cv2.INPAINT_TELEA)
            x = Image.fromarray(x)

        else:

            nanmask = np.isnan(x)

            if not np.any(nanmask):
                return x

            tx = x[:, :, 0]
            tx = cv2.inpaint(tx, nanmask[:, :, 0].astype(np.uint8), 2, cv2.INPAINT_TELEA)

            ty = x[:, :, 1]
            ty = cv2.inpaint(ty, nanmask[:, :, 1].astype(np.uint8), 2, cv2.INPAINT_TELEA)

            x = np.stack((tx, ty), axis=-1)

        return x

class RandomOrderFlip(object):

    def __init__(self, p=0.5):
        self.p = p
        self.draw()

    def draw(self):
        self.flip = np.random.random()

    def __call__(self, image1, image2):

        #
        self.draw()

        if self.flip < self.p:
            return image2, image1, -1
        else:
            return image1, image2, 1

class RandomHorizontalFlip(object):

    def __init__(self, p=0.5):
        self.p = p
        self.draw()

    def draw(self):
        self.flip = np.random.random()

    def __call__(self, x):

        if self.flip < self.p:
            if isinstance(x, Image.Image):
                x = F.hflip(x)
            else:
                x = np.fliplr(x)
                x = x + 0 # hack
                x[:, :, 0] = -x[:, :, 0] # negate horizontal displacement
        return x

class ScaledCenterCrop(object):

    def __init__(self, input_size, output_size):

        w, h = input_size
        tw, th = output_size

        self.i = int(round((h - th) / 2.))
        self.j = int(round((w - tw) / 2.))

        self.w = tw
        self.h = th

    def new_params(self):
        return None

    def correct_calibration(self, K, scale):

        dx = self.j * scale
        dy = self.i * scale

        K[0, -1] = K[0, -1] - dx
        K[1, -1] = K[1, -1] - dy

        return K

    def __call__(self, x, scale):

        i = self.i * scale
        j = self.j * scale
        h = self.h * scale
        w = self.w * scale

        if isinstance(x, Image.Image):
            # PIL image
            return F.crop(x, i, j, h, w)
        else:
            # numpy array
            return x[i:i+h, j:j+w]

class ScaledRandomCrop(object):

    def __init__(self, input_size, output_size):

        self.input_size = input_size
        self.output_size = output_size

        self.new_params()

    def get_params(self, input_size, output_size):
        w, h = input_size
        tw, th = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        elif w == tw and h > th:
            i = np.random.randint(0, h-th)
            return i, 0, th, tw
        elif w > tw and h == th:
            j = np.random.randint(0, w-tw)
            return 0, j, th, tw
        else:
            i = np.random.randint(0, h-th)
            j = np.random.randint(0, w-tw)
            return i, j, th, tw

    def new_params(self):
        self.i, self.j, self.h, self.w = self.get_params(self.input_size, self.output_size)

    def correct_calibration(self, K, scale):

        dx = self.j * scale
        dy = self.i * scale

        K[0, -1] = K[0, -1] - dx
        K[1, -1] = K[1, -1] - dy

        return K

    def __call__(self, x, scale):

        i = self.i * scale
        j = self.j * scale
        h = self.h * scale
        w = self.w * scale

        if isinstance(x, Image.Image):
            # PIL image
            return F.crop(x, i, j, h, w)
        else:
            # numpy array
            return x[i:i+h, j:j+w]

class RandomCrop(object):

    def __init__(self, input_size, output_size):

        self.input_size = input_size
        self.output_size = output_size

        self.new_params()

    def get_params(self, input_size, output_size):
        w, h = input_size
        tw, th = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        elif w == tw and h > th:
            i = np.random.randint(0, h-th)
            return i, 0, th, tw
        elif w > tw and h == th:
            j = np.random.randint(0, w-tw)
            return 0, j, th, tw
        else:
            i = np.random.randint(0, h-th)
            j = np.random.randint(0, w-tw)
            return i, j, th, tw

    def new_params(self):
        self.i, self.j, self.h, self.w = self.get_params(self.input_size, self.output_size)

    def __call__(self, x):

        if isinstance(x, Image.Image):
            # PIL image
            return F.crop(x, self.i, self.j, self.h, self.w)
        else:
            # numpy array
            return x[self.i:self.i+self.h, self.j:self.j+self.w]

class CenterCrop(object):

    def __init__(self, input_size, output_size):

        w, h = input_size
        tw, th = output_size

        self.i = int(round((h - th) / 2.))
        self.j = int(round((w - tw) / 2.))

        self.w = tw
        self.h = th

    def new_params(self):
        return None

    def __call__(self, x):

        if isinstance(x, Image.Image):
            # PIL image
            return F.crop(x, self.i, self.j, self.h, self.w)
        else:
            # numpy array
            return x[self.i:self.i+self.h, self.j:self.j+self.w]

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def flowshow(flow):
    # npflow = flow.numpy()
    # npflow = np.transpose(npflow, (1, 2, 0))
    img = flow_utils.compute_flow_image(flow)
    plt.imshow(img)
    plt.show()

def write_flow(name, flow):
    with open(name, 'wb') as f:
        f.write('PIEH'.encode('utf-8'))
        np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
        flow = flow.astype(np.float32)
        flow.tofile(f)

def flow_loader(path):
    TAG_FLOAT = 202021.25
    with open(path, "rb") as f:
        tag = np.fromfile(f, np.float32, count=1)
        assert tag[0] == TAG_FLOAT, 'Flow number %r incorrect. Invalid .flo file' % tag
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2*w*h)
        flow = np.reshape(data, (int(h), int(w), 2))
        return flow

def pfm_loader(path):
    file = open(path, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip().decode("utf-8")
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("utf-8"))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip().decode("utf-8"))
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale

def flow_png_loader(path):
    flow_raw = cv2.imread(path, -1)

    # convert BGR to RG
    flow = flow_raw[:, :, 2:0:-1].astype(np.float32)

    # scaling
    flow = flow - 32768.
    flow = flow / 64.

    # clip
    flow[np.abs(flow) < 1e-10] = 1e-10

    # invalid mask
    invalid_mask = (flow_raw[:, :, 0] == 0)
    flow[invalid_mask, :] = 0

    # valid mask
    valid_mask = (flow_raw[:, :, 0] == 1).astype(np.uint8)

    return flow, valid_mask

class Sintel(Dataset):

    def __init__(self, root, split='train', passes=['final'], data_augmentation=True,
            transform=None, target_transform=None, pyramid_levels=[0], 
            flow_scale=1, crop_dim=(384, 768), hflip=-1):
        super(Sintel, self).__init__()

        pyramid_levels = sorted(pyramid_levels)

        if split == 'test':
            root = os.path.join(root, 'test')
        else:
            root = os.path.join(root, 'training')

        self.transform = transform
        self.target_transform = target_transform

        if data_augmentation:
            self.flip_transform = RandomHorizontalFlip(hflip)
            self.crop_transform = RandomCrop(SINTEL_DIM, crop_dim[::-1])
        else:
            self.flip_transform = RandomHorizontalFlip(-1)
            self.crop_transform = CenterCrop(SINTEL_DIM, crop_dim[::-1])

        dim = (crop_dim[0] // 2**pyramid_levels[0], crop_dim[1] // 2**pyramid_levels[0])
        self.first_level_resize = transforms.Resize(dim)

        self.pyramid_transforms = []
        for l in pyramid_levels:
            dim = (crop_dim[0] // 2**l, crop_dim[1] // 2**l)
            self.pyramid_transforms.append(flow_utils.ResizeFlow(dim))
        self.scale_transform = flow_utils.ScaleFlow(flow_scale)

        passdirs = [os.path.join(root, p) for p in passes]
        self.dataset = list(itertools.chain(*[self.make_dataset(p) for p in passdirs]))

    def __getitem__(self, idx):

        image1_path, image2_path, flow_path = self.dataset[idx]
        image1 = pil_loader(image1_path)
        image2 = pil_loader(image2_path)

        image1 = self.crop_transform(image1)
        image2 = self.crop_transform(image2)

        image1 = self.flip_transform(image1)
        image2 = self.flip_transform(image2)

        image1 = self.first_level_resize(image1)
        image2 = self.first_level_resize(image2)

        if self.transform is not None:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        flow0 = flow_loader(flow_path)

        flow0 = self.crop_transform(flow0)
        flow0 = self.flip_transform(flow0)

        flow0 = self.scale_transform(flow0)
        flow_levels = []
        for pt in self.pyramid_transforms: 
            flow = pt(flow0) 
            flow_levels.append(flow)

        if self.target_transform is not None:
            flow_levels = [self.target_transform(flow) for flow in flow_levels]

        self.crop_transform.new_params()
        self.flip_transform.draw()

        return image1, image2, flow_levels

    def __len__(self):
        return len(self.dataset)

    def make_dataset(self, passdir):
        dataset = []
        flowdir = os.path.join(os.path.dirname(passdir), 'flow')
        for seqid in sorted(os.listdir(passdir)):
            seqdir = os.path.join(passdir, seqid)
            for sd, _, fnames in sorted(os.walk(seqdir)):
                for f1, f2 in zip(sorted(fnames), sorted(fnames)[1:]):
                    image1 = os.path.join(sd, f1)
                    image2 = os.path.join(sd, f2)
                    flow = os.path.join(flowdir, seqid, f1.split('.')[0] + '.flo')
                    dataset.append((image1, image2, flow))
        return dataset

class SintelSR(Dataset):
    
    def __init__(self, root, split='train', passes=['final'], transform=None, 
            input_scale=2, target_scale=1, crop_dim=(384, 768)):
        super(SintelSR, self).__init__()

        if split == 'test':
            root = os.path.join(root, 'test')
        else:
            root = os.path.join(root, 'training')

        self.transform = transform

        self.crop_transform = transforms.RandomCrop(crop_dim)

        self.input_resize = None
        if input_scale != 1:
            input_dim = (crop_dim[0] // input_scale, crop_dim[1] // input_scale)
            self.input_resize = transforms.Resize(input_dim)

        self.target_resize = None
        if target_scale != 1:
            target_dim = (crop_dim[0] // target_scale, crop_dim[1] // target_scale)
            self.target_resize = transforms.Resize(target_dim)

        self.tensor_transform = transforms.ToTensor()

        passdirs = [os.path.join(root, p) for p in passes]
        self.dataset = list(itertools.chain(*[self.make_dataset(p) for p in passdirs]))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        image_path = self.dataset[idx]
        image = pil_loader(image_path)
        image = self.crop_transform(image)

        input_image = image
        target_image = image

        if self.input_resize is not None:
            input_image = self.input_resize(input_image)

        if self.target_resize is not None:
            target_image = self.target_resize(target_image)

        if self.transform is not None:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)

        input_image = self.tensor_transform(input_image)
        target_image = 2 * self.tensor_transform(target_image) - 1

        return input_image, target_image

    def make_dataset(self, passdir):
        dataset = []
        for seqid in sorted(os.listdir(passdir)):
            seqdir = os.path.join(passdir, seqid)
            fnames = sorted(glob.glob(seqdir + '/*.png'))
            dataset.extend(fnames)
        return dataset

class FlyingChairs(Dataset):

    def __init__(self, root, transform=None, target_transform=None, 
            pyramid_levels=[0], flow_scale=1, crop_dim=(384, 448)):
        super(FlyingChairs, self).__init__()

        pyramid_levels = sorted(pyramid_levels)

        self.transform = transform
        self.target_transform = target_transform

        self.crop_transform = RandomCrop(CHAIRS_DIM, crop_dim[::-1])

        dim = (crop_dim[0] // 2**pyramid_levels[0], crop_dim[1] // 2**pyramid_levels[0])
        self.first_level_resize = transforms.Resize(dim)

        self.pyramid_transforms = []
        for l in pyramid_levels:
            dim = (crop_dim[0] // 2**l, crop_dim[1] // 2**l)
            self.pyramid_transforms.append(flow_utils.ResizeFlow(dim))
        self.scale_transform = flow_utils.ScaleFlow(flow_scale)

        self.dataset = self.make_dataset(root)

    def __getitem__(self, idx):

        image1_path, image2_path, flow_path = self.dataset[idx]
        image1 = pil_loader(image1_path)
        image2 = pil_loader(image2_path)

        image1 = self.crop_transform(image1)
        image2 = self.crop_transform(image2)

        image1 = self.first_level_resize(image1)
        image2 = self.first_level_resize(image2)

        if self.transform is not None:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        flow0 = flow_loader(flow_path)

        flow0 = self.crop_transform(flow0)

        flow0 = self.scale_transform(flow0)
        flow_levels = []
        for pt in self.pyramid_transforms: 
            flow = pt(flow0) 
            flow_levels.append(flow)

        if self.target_transform is not None:
            flow_levels = [self.target_transform(flow) for flow in flow_levels]

        self.crop_transform.new_params()

        return image1, image2, flow_levels

    def __len__(self):
        return len(self.dataset)

    def make_dataset(self, root):

        fnames = sorted(glob.glob(root + '/*.ppm'))
        images1 = fnames[::2]
        images2 = fnames[1::2]
        flows = [f.split('img')[0] + 'flow.flo' for f in images1]
        dataset = list(zip(images1, images2, flows))
        return dataset

class FlyingChairsSR(Dataset):
    
    def __init__(self, root, transform=None, 
            input_scale=2, target_scale=1, crop_dim=(384, 448)):
        super(FlyingChairsSR, self).__init__()

        self.transform = transform

        self.crop_transform = transforms.RandomCrop(crop_dim)

        self.input_resize = None
        if input_scale != 1:
            input_dim = (crop_dim[0] // input_scale, crop_dim[1] // input_scale)
            self.input_resize = transforms.Resize(input_dim)

        self.target_resize = None
        if target_scale != 1:
            target_dim = (crop_dim[0] // target_scale, crop_dim[1] // target_scale)
            self.target_resize = transforms.Resize(target_dim)

        self.tensor_transform = transforms.ToTensor()

        self.dataset = self.make_dataset(root)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        image_path = self.dataset[idx]
        image = pil_loader(image_path)
        image = self.crop_transform(image)

        input_image = image
        target_image = image

        if self.input_resize is not None:
            input_image = self.input_resize(input_image)

        if self.target_resize is not None:
            target_image = self.target_resize(target_image)

        if self.transform is not None:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)

        input_image = self.tensor_transform(input_image)
        target_image = 2 * self.tensor_transform(target_image) - 1

        return input_image, target_image

    def make_dataset(self, root):
        return sorted(glob.glob(root + '/*.ppm'))

class FlyingThings(Dataset):

    def __init__(self, root, split='train', partition=['A', 'B', 'C'], transform=None, target_transform=None,
            pyramid_levels=[0], flow_scale=1, crop_dim=(384, 768)):
        super(FlyingThings, self).__init__()

        pyramid_levels = sorted(pyramid_levels)

        self.transform = transform
        self.target_transform = target_transform

        self.nanfilter_transform = InpaintNaNs()

        self.crop_transform = RandomCrop(THINGS_DIM, crop_dim[::-1])

        dim = (crop_dim[0] // 2**pyramid_levels[0], crop_dim[1] // 2**pyramid_levels[0])
        self.first_level_resize = transforms.Resize(dim)

        self.pyramid_transforms = []
        for l in pyramid_levels:
            dim = (crop_dim[0] // 2**l, crop_dim[1] // 2**l)
            self.pyramid_transforms.append(flow_utils.ResizeFlow(dim))
        self.scale_transform = flow_utils.ScaleFlow(flow_scale)

        self.dataset = self.make_dataset(root, split, partition)

    def __getitem__(self, idx):

        image1_path, image2_path, flow_path = self.dataset[idx]
        image1 = pil_loader(image1_path)
        image2 = pil_loader(image2_path)

        image1 = self.nanfilter_transform(image1)
        image2 = self.nanfilter_transform(image2)

        image1 = self.crop_transform(image1)
        image2 = self.crop_transform(image2)

        image1 = self.first_level_resize(image1)
        image2 = self.first_level_resize(image2)

        if self.transform is not None:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        flow0, _ = pfm_loader(flow_path) # add pfm loader
        flow0 = flow0[:, :, :2]

        flow0 = self.nanfilter_transform(flow0)

        flow0 = self.crop_transform(flow0)

        flow0 = self.scale_transform(flow0)
        flow_levels = []
        for pt in self.pyramid_transforms: 
            flow = pt(flow0) 
            flow_levels.append(flow)

        if self.target_transform is not None:
            flow_levels = [self.target_transform(flow) for flow in flow_levels]

        self.crop_transform.new_params()

        return image1, image2, flow_levels

    def __len__(self):
        return len(self.dataset)

    def make_dataset(self, root, split, partition):

        image_pairs = self.image_paths(root, split, partition)
        image1, image2 = zip(*image_pairs)
        flows = self.flow_paths(root, split, partition)
        dataset = list(zip(image1, image2, flows))

        return dataset

    def image_paths(self, root, split, partition):

        root = os.path.join(root, 'frames_finalpass')
        if split == 'test':
            root = os.path.join(root, 'TEST')
        else:
            root = os.path.join(root, 'TRAIN')

        image_pairs = []
        for part in partition:
            part_path = os.path.join(root, part)
            for subseq_path in sorted(glob.glob(part_path + '/*')):
                # future direction
                for camera in ['left', 'right']:
                    camera_path = os.path.join(subseq_path, camera)
                    fnames = sorted(glob.glob(camera_path + '/*.png'))
                    subseq_pairs = list(zip(fnames, fnames[1:]))
                    image_pairs.extend(subseq_pairs)
                # past direction
                for camera in ['left', 'right']:
                    camera_path = os.path.join(subseq_path, camera)
                    fnames = sorted(glob.glob(camera_path + '/*.png'))
                    subseq_pairs = list(zip(fnames[1:], fnames))
                    image_pairs.extend(subseq_pairs)

        return image_pairs

    def flow_paths(self, root, split, partition):

        root = os.path.join(root, 'optical_flow')
        if split == 'test':
            root = os.path.join(root, 'TEST')
        else:
            root = os.path.join(root, 'TRAIN')

        flows = []
        for part in partition:
            part_path = os.path.join(root, part)
            for subseq_path in sorted(glob.glob(part_path + '/*')):
                direction_path = os.path.join(subseq_path, 'into_future')
                for camera in ['left', 'right']:
                    camera_path = os.path.join(direction_path, camera)
                    fnames = sorted(glob.glob(camera_path + '/*.pfm'))
                    flows.extend(fnames[:-1])
                direction_path = os.path.join(subseq_path, 'into_past')
                for camera in ['left', 'right']:
                    camera_path = os.path.join(direction_path, camera)
                    fnames = sorted(glob.glob(camera_path + '/*.pfm'))
                    flows.extend(fnames[1:])

        return flows

class KITTIFlow(Dataset):

    def __init__(self, root, split='train', transform=None, target_transform=None, 
            pyramid_levels=[0], flow_scale=1, crop_dim=(320, 896)):
        super(KITTIFlow, self).__init__()

        pyramid_levels = sorted(pyramid_levels)

        self.split = split
        if self.split == 'test':
            root = os.path.join(root, 'testing')
        else:
            root = os.path.join(root, 'training')

        self.transform = transform
        self.target_transform = target_transform

        self.flow_to_tensor = flow_utils.ToTensor()
        self.mask_to_tensor = MaskToTensor()

        self.crop_transform = RandomCrop(KITTI_DIM, crop_dim[::-1])

        dim = (crop_dim[0] // 2**pyramid_levels[0], crop_dim[1] // 2**pyramid_levels[0])
        self.first_level_resize = transforms.Resize(dim)

        self.scale_transform = flow_utils.ScaleFlow(flow_scale)

        self.pyramid_transforms = []
        for l in pyramid_levels:
            self.pyramid_transforms.append(flow_utils.ResizeSparseFlow(2**l))

        self.dataset = self.make_dataset(root, split)

    def __getitem__(self, idx):

        image1_path, image2_path, flow_path = self.dataset[idx]
        image1 = pil_loader(image1_path)
        image2 = pil_loader(image2_path)
        
        # NOTE: Hack for dealing with different sizes between samples
        self.crop_transform.input_size = image1.size
        self.crop_transform.new_params()

        image1 = self.crop_transform(image1)
        image2 = self.crop_transform(image2)

        image1 = self.first_level_resize(image1)
        image2 = self.first_level_resize(image2)

        if self.transform is not None:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        if self.split == 'test':
            return image1, image2

        flow0, valid_mask0 = flow_png_loader(flow_path)

        flow0 = self.crop_transform(flow0)
        valid_mask0 = self.crop_transform(valid_mask0)

        flow0 = self.scale_transform(flow0)

        flow_levels = []
        mask_levels = []
        for pt in self.pyramid_transforms: 
            flow, mask  = pt(flow0, valid_mask0) 
            flow_levels.append(flow)
            mask_levels.append(mask)

        flow_levels = [self.flow_to_tensor(flow) for flow in flow_levels]
        mask_levels = [self.mask_to_tensor(mask) for mask in mask_levels]

        if self.target_transform is not None:
            flow_levels = [self.target_transform(flow) for flow in flow_levels]

        return image1, image2, flow_levels, mask_levels

    def __len__(self):
        return len(self.dataset)

    def make_dataset(self, root, split):

        imagedir = os.path.join(root, 'image_2')
        fnames = sorted(glob.glob(imagedir + '/*.png'))
        images1 = fnames[::2]
        images2 = fnames[1::2]

        flowdir = os.path.join(root, 'flow_noc')
        flows = sorted(glob.glob(flowdir + '/*.png'))

        dataset = []
        if split == 'test':
            dataset = list(zip(images1, images2))
        else:
            dataset = list(zip(images1, images2, flows))

        return dataset

class KITTIFlowSR(Dataset):
    
    def __init__(self, root, split='train', transform=None, 
            input_scale=2, target_scale=1, crop_dim=(320, 896)):
        super(KITTIFlowSR, self).__init__()

        if split == 'test':
            root = os.path.join(root, 'test')
        else:
            root = os.path.join(root, 'training')

        self.transform = transform

        self.crop_transform = transforms.RandomCrop(crop_dim)

        self.input_resize = None
        if input_scale != 1:
            input_dim = (crop_dim[0] // input_scale, crop_dim[1] // input_scale)
            self.input_resize = transforms.Resize(input_dim)

        self.target_resize = None
        if target_scale != 1:
            target_dim = (crop_dim[0] // target_scale, crop_dim[1] // target_scale)
            self.target_resize = transforms.Resize(target_dim)

        self.tensor_transform = transforms.ToTensor()

        self.dataset = self.make_dataset(root)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        image_path = self.dataset[idx]
        image = pil_loader(image_path)
        image = self.crop_transform(image)

        input_image = image
        target_image = image

        if self.input_resize is not None:
            input_image = self.input_resize(input_image)

        if self.target_resize is not None:
            target_image = self.target_resize(target_image)

        if self.transform is not None:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)

        input_image = self.tensor_transform(input_image)
        target_image = 2 * self.tensor_transform(target_image) - 1

        return input_image, target_image

    def make_dataset(self, root):
        imagedir = os.path.join(root, 'image_2')
        dataset = sorted(glob.glob(imagedir + '/*.png'))
        return dataset

class KITTIDerot(Dataset):

    def __init__(self, root, sequences=[0, 1, 2, 3, 4, 5], transform=None, input_scale=None, frame_offset=None,
            pyramid_levels=[0], crop_dim=(20, 56), data_augmentation=True, fflip=-1, return_id=False):
        super(KITTIDerot, self).__init__()

        if frame_offset is None:
            frame_offset = [1, 2, 3]

        self.return_id = return_id

        self.pyramid_levels = sorted(pyramid_levels)

        if input_scale is None:
            input_scale = self.pyramid_levels[-1]
        self.input_scale = input_scale

        self.transform = transform

        if data_augmentation:
            self.flow_flip_transform = RandomOrderFlip(fflip)
            self.crop_transform = ScaledRandomCrop(KITTI_COARSE_DIM, crop_dim[::-1])
        else:
            self.flow_flip_transform = RandomOrderFlip(-1)
            self.crop_transform = ScaledCenterCrop(KITTI_COARSE_DIM, crop_dim[::-1])

        self.flow_to_tensor = flow_utils.ToTensor()
        self.mask_to_tensor = MaskToTensor()

        self.dataset = self.make_dataset(root, sequences, self.pyramid_levels, input_scale, frame_offset)

    def __getitem__(self, idx):

        image1_path, image2_path, mask_path, calib_paths, t, subseq_id, pair_id = self.dataset[idx]
        image1 = pil_loader(image1_path)
        image2 = pil_loader(image2_path)

        mask = kitti_invalid_mask(pil_loader(mask_path))
        
        image1, image2, sign_flip = self.flow_flip_transform(image1, image2)
        t = sign_flip * t

        Ks = []
        for cp in calib_paths:
            Ks.append(np.fromfile(cp, sep=' ').reshape((3, 3)))
        Ks = Ks[::-1]

        flow_levels = []
        for l in range(len(self.pyramid_levels)):

            flow_shape = (image1.size[1] * 2**l, image1.size[0] * 2**l, 2)

            if np.linalg.norm(t) < np.finfo(np.float32).eps:
                flow = np.zeros(flow_shape)
                flow_levels.append(flow)
                continue

            # calibration
            K = Ks[l]

            # epipole
            e = np.dot(K, t)
            e = e / (e[2] + np.finfo(np.float32).eps)
            e = e[:2]

            # epipole matrix
            E = e[None, None, :]
            E = np.tile(E, (flow_shape[0], flow_shape[1], 1))

            # init pixel map
            px, py = np.meshgrid(np.arange(flow_shape[1]), np.arange(flow_shape[0]))
            X = np.stack((px, py), axis=2)

            # flow map
            flow = E - X
            fmag = np.sqrt(np.sum(np.multiply(flow, flow), axis=2))
            flow = np.divide(flow, np.stack((fmag, fmag), axis=2))

            # flip if forward translation
            flow = -np.sign(t[2]) * flow

            # append
            flow_levels.append(flow)

        # apply crops
        self.crop_transform.new_params()
        image1 = self.crop_transform(image1, 2**(4-self.input_scale))
        image2 = self.crop_transform(image2, 2**(4-self.input_scale))
        mask = self.crop_transform(mask, 2**(4-self.input_scale))
        Ks = [self.crop_transform.correct_calibration(K, 2**l) for l, K in enumerate(Ks)]
        flow_levels = [self.crop_transform(flow, 2**l) for l, flow in enumerate(flow_levels)]
        flow_levels = flow_levels[::-1]

        # convert to tensor
        flow_levels = [self.flow_to_tensor(flow) for flow in flow_levels]
        mask = self.mask_to_tensor(mask)
        image1 = F.to_tensor(image1)
        image2 = F.to_tensor(image2)

        # remove blue pixels
        if sign_flip < 0:
            image1[:, mask[0]] = 0
        else:
            image2[:, mask[0]] = 0

        # optional input image transform
        if self.transform is not None:
            image1 = F.to_tensor(self.transform(F.to_pil_image(image1)))
            image2 = F.to_tensor(self.transform(F.to_pil_image(image2)))

        if self.return_id:
            return image1, image2, flow_levels, Ks[-1], t, subseq_id, pair_id
        else:
            return image1, image2, flow_levels, Ks[-1], t

    def __len__(self):
        return len(self.dataset)

    def make_dataset(self, root, sequences, pyramid_levels, input_scale, frame_offset):

        input_scale = 2**input_scale

        window_fids = [0] + frame_offset

        dataset = []
        for s in sequences:
            scaledir = os.path.join(root, "%02d" % s, 'subsequences', "%dx" % input_scale)
            subseqdirs = sorted(glob.glob(scaledir + '/*/'))
            for subseq in subseqdirs:

                # all images in folder
                fnames = sorted(glob.glob(subseq + '/*.png'))
                nimages = len(fnames)

                # select relevant frames
                fnames = [fnames[i] for i in window_fids]
                npairs = len(fnames[1:])

                fmasks = fnames[1:]

                ftrans = os.path.join(subseq, 'translations.txt')
                translations = np.fromfile(ftrans, sep=' ').reshape((nimages,3))

                fcalibs = []
                for f in fnames[1:]:
                    pcalibs = []
                    for p in pyramid_levels:
                        pyrdir = os.path.join(root, "%02d" % s, 'subsequences', "%dx" % 2**p)
                        refdir = os.path.join(pyrdir, subseq.split('/')[-2])
                        pcalibs.append(os.path.join(refdir, 'scaled_calibration.txt'))
                    fcalibs.append(tuple(pcalibs))

                subseq_id = os.path.basename(os.path.dirname(fnames[0]))
                pair_ids = ['0{:d}'.format(i) for i in frame_offset]

                pairs = list(zip([fnames[0]]*npairs, fnames[1:], fmasks, fcalibs, 
                    translations[frame_offset, :], [subseq_id]*npairs, pair_ids))

                dataset.extend(pairs)

        return dataset
