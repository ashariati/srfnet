import sys

import torch
import torch.nn as nn

import torchvision
import torchvision.transforms.functional as F

import numpy as np
import scipy as sp
import scipy.interpolate

import ransac

import pdb

def valid_flow_mask(flow, mask):

    batch_size = flow.shape[0]
    xmax = flow.shape[-1]
    ymax = flow.shape[-2]

    px, py = np.meshgrid(np.arange(xmax), np.arange(ymax))
    px = torch.from_numpy(px[None, :, :]).float().to(flow.device)
    py = torch.from_numpy(py[None, :, :]).float().to(flow.device)

    dx = px + flow[:, 0, :, :]
    dx = torch.round(dx[:, None, :, :])
    dy = py + flow[:, 1, :, :]
    dy = torch.round(dy[:, None, :, :])

    dx = torch.where(dx < xmax, dx, (xmax-1) * torch.ones(dx.shape).to(flow.device))
    dx = torch.where(dx > 0, dx, torch.zeros(dx.shape).to(flow.device)).long()

    dy = torch.where(dy < ymax, dy, (ymax-1) * torch.ones(dy.shape).to(flow.device))
    dy = torch.where(dy > 0, dy, torch.zeros(dy.shape).to(flow.device)).long()

    valid_mask = torch.zeros(mask.shape).byte().to(mask.device)
    for i in range(batch_size):
        valid_mask[i] = mask[i, 0, dy[i, 0, :, :], dx[i, 0, :, :]]

    return valid_mask

def dotprod_clamp(x, eps=1e-7):
    return torch.clamp(x, -1+eps, 1-eps)

def acos_safe(x, eps=1e-7):
    sign = torch.sign(x)
    slope = np.arccos(1-eps) / eps
    return torch.where(torch.abs(x) <= 1-eps,
            torch.acos(x),
            torch.acos(sign * (1 - eps)) - slope*sign*(torch.abs(x) - 1 + eps))

class EpipolarLoss(nn.Module):
    def __init__(self, percentile, thresh):
        super(EpipolarLoss, self).__init__()

        self.percentile = percentile
        self.thresh = thresh

    def forward(self, y_hat, y):

        # normalize
        mag = torch.norm(y_hat, p=2, dim=1, keepdim=True)
        y_norm = torch.div(y_hat, mag)

        # compute orientation
        dp = torch.sum(torch.mul(y_norm, y), dim=1, keepdim=True)
        dp = dotprod_clamp(dp)
        theta = acos_safe(dp)

        # compute percentile 
        batch_size = y.shape[0]
        ts = torch.reshape(theta, (batch_size, -1,))
        vals, indices = torch.sort(ts, dim=1)
        pid = int(ts.shape[1] * self.percentile)
        percmask = torch.zeros(ts.shape[1]).byte().to(vals.device)
        percmask[:pid] = 1

        # create scene mask
        mask_shape = list(y.size())
        mask_shape[1] = 1
        scene_mask = torch.zeros(mask_shape).byte().to(y.device)
        for i in range(batch_size):
            smv = scene_mask.view((batch_size, -1,))
            if vals[i, pid].item() < self.thresh:
                smv[i, indices[i, percmask]] = 1
            else:
                smv[i, indices[i, vals[i, :] < self.thresh]] = 1

        return torch.masked_select(theta, scene_mask).sum()

def trans_error(flow, K, t, debug=False):

    if isinstance(t, torch.Tensor):
        t = t.numpy()

    batch_size = flow.shape[0]

    if flow.is_cuda:
        flow = flow.cpu()

    # inlier mask
    in_mask = torch.zeros((batch_size, 1, flow.shape[2], flow.shape[3])).byte()

    # error image
    error_image = torch.zeros((batch_size, 1, flow.shape[2], flow.shape[3])).float()

    # pixel map
    px, py = np.meshgrid(np.arange(flow.shape[-1]), np.arange(flow.shape[-2]))

    epe = []
    theta = []
    pcent = []
    translations = []
    n = flow.shape[-1] * flow.shape[-2]
    model = ransac.EpipoleModel()
    for i in range(batch_size):

        # translation scale
        ti = t[i]
        scale = np.linalg.norm(ti)

        # find epipole
        flow_numpy = to_flow(flow[i])
        data = np.concatenate((flow_numpy, px[:, :, None], py[:, :, None]), axis=2)
        data = np.reshape(data.transpose((2, 0, 1)), (4, -1))
        e, inliers = ransac.ransac(data.T, model, 2, 500, 2)

        # mask of inliers 
        inpx = data[2, inliers['inliers_idxs']].astype(np.int32)
        inpy = data[3, inliers['inliers_idxs']].astype(np.int32)
        in_mask[i, 0, inpy, inpx] = 1

        # error image
        errors = model.get_error(data.T, e)
        error_image[i, 0, :, :] = torch.from_numpy(errors.reshape(flow.shape[2:]))

        # make homogeneous
        e = np.ravel(e)
        e_tilde = np.append(e, 1)

        # compute translation vector
        t_img = np.dot(np.linalg.inv(K[i]), e_tilde)
        t_mag = np.linalg.norm(t_img)
        t_hat = t_img * (scale / t_mag)

        # angular difference with ground truth
        dp = np.dot(t_img / t_mag, ti / scale)
        flip = np.sign(dp)
        dp = flip * dp
        t_hat = flip * t_hat

        # save
        epe.append(np.linalg.norm(t_hat - ti))
        theta.append(np.arccos(dp) * (180. / np.pi))
        pcent.append(100. * (epe[-1] / scale))
        translations.append(t_hat)

    if debug:
        return np.array(epe), np.array(pcent), np.array(theta), translations, in_mask, error_image
    else:
        return np.array(epe), np.array(pcent), np.array(theta)

def flow_to_line(flow):

    # pixel map
    px, py = np.meshgrid(np.arange(flow.shape[-1]), np.arange(flow.shape[-2]))
    px = torch.from_numpy(px).float().to(flow.device)
    py = torch.from_numpy(py).float().to(flow.device)

    # normalize flow
    flow_mag = torch.norm(flow, p=2, dim=1, keepdim=True)
    flow = torch.div(flow, flow_mag)

    # line direction
    line = torch.flip(flow, [1])
    line[:, 1, :, :] = -line[:, 1, :, :]

    # constant
    C = torch.mul(line[:, 0, :, :], px) + torch.mul(line[:, 1, :, :], py)
    C = C[:, None, :, :]

    return torch.cat((line, C), dim=1) 


class MaskedRobustLoss(nn.Module):
    def __init__(self, eps, q):
        super(MaskedRobustLoss, self).__init__()

        self.eps = eps
        self.q = q

    def forward(self, y, y_hat, mask):
        error = torch.pow((torch.sum(torch.abs(y - y_hat), dim=1, keepdim=True) + self.eps), self.q)
        masked_error = torch.masked_select(error, mask)
        return masked_error.sum()

class RobustLoss(nn.Module):
    def __init__(self, eps, q):
        super(RobustLoss, self).__init__()

        self.eps = eps
        self.q = q

    def forward(self, y, y_hat):
        error = torch.pow((torch.sum(torch.abs(y - y_hat), dim=1) + self.eps), self.q).sum()
        return error

class EPELoss(nn.Module):
    def __init__(self):
        super(EPELoss, self).__init__()

    def forward(self, y, y_hat):
        return (y - y_hat).pow(2).sum(dim=1).sqrt().sum()

def MaskedAEPE(Y, Y_hat, mask):
    error = torch.sqrt(torch.sum(torch.pow(Y - Y_hat, 2), dim=1, keepdim=True))
    masked_error = torch.masked_select(error, mask)
    return masked_error.mean()

def AEPE(Y, Y_hat):
    return (Y - Y_hat).pow(2).sum(dim=1).sqrt().mean()

class ScaleFlow(object):

    def __init__(self, scale):
        self.scale = scale

    def __call__(self, sample):
        return scale_flow(sample, self.scale)

def scale_flow(flow, scale):
    return flow / scale

class ResizeSparseFlow(object):

    def __init__(self, scale):
        assert isinstance(scale, int)
        self.scale = scale

    def __call__(self, sample, mask):
        return resize_sparse_flow(sample, mask, self.scale)

def resize_sparse_flow(flow, mask, scale):

    if scale == 1:
        return flow, mask

    h, w = flow.shape[:2]
    ow = w // scale
    oh = h // scale

    xx, yy = np.meshgrid(np.arange(ow), np.arange(oh))
    xx = xx * scale
    yy = yy * scale

    I = np.zeros((oh, ow, scale*scale), dtype=np.int32)
    J = np.zeros((oh, ow, scale*scale), dtype=np.int32)
    for i in range(scale*scale):
        I[:, :, i] = yy + (i // scale)
        J[:, :, i] = xx + (i % scale)

    Fx = flow[I, J, 0]
    Fy = flow[I, J, 1]

    M = mask[I, J]
    W = np.sum(M, axis=2)

    out_mask = (W != 0).astype(np.uint8)
    W[W == 0] = np.iinfo(W.dtype).max

    fx = np.divide(np.sum(Fx, axis=2), W)
    fy = np.divide(np.sum(Fy, axis=2), W)

    out_flow = np.stack((fx, fy), axis=2)

    return out_flow, out_mask

class ResizeFlow(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        return resize_flow(sample, self.output_size)

def resize_flow(flow, output_size):

    h, w = flow.shape[:2]
    if output_size == (h, w):
        return flow

    ow = oh = None
    if isinstance(output_size, int):
        if w < h:
            ow = output_size
            oh = int(output_size * h / w)
        else:
            oh = output_size
            ow = int(output_size * w / h)
    else:
        oh, ow = output_size

    x = np.arange(w)
    y = np.arange(h)
    fx = sp.interpolate.interp2d(x, y, flow[:,:,0])
    fy = sp.interpolate.interp2d(x, y, flow[:,:,1])

    ox = np.linspace(0, w, ow)
    oy = np.linspace(0, h, oh)

    Fx = fx(ox, oy)
    Fy = fy(ox, oy)

    out_flow = np.stack((Fx, Fy), axis=2)

    return out_flow

class ToTensor(object):

    def __call__(self, sample):
        return to_tensor(sample)

def to_tensor(flow):
    return torch.from_numpy(flow.transpose((2, 0, 1))).float()

class ToPILImage(object):

    def __call__(self, sample):
        return to_pil_image(sample)

def to_pil_image(tensor):
    return F.to_pil_image(compute_flow_image(to_flow(tensor)))

class ToFlow(object):

    def __call__(self, sample):
        return to_flow(sample)

def to_flow(tensor):
    return tensor.numpy().transpose((1, 2, 0))

class ToRGBImage(object):

    def __call__(self, sample):
        return compute_flow_image(sample)

def make_color_wheel():

    #  color encoding scheme
    
    #   adapted from the color circle idea described at
    #   http://members.shaw.ca/quadibloc/other/colint.htm
    
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6
    
    ncols = RY + YG + GC + CB + BM + MR
    
    colorwheel = np.zeros([ncols, 3]) # r g b
    
    col = 0
    #RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0, RY, 1)/RY)
    col += RY
    
    #YG
    colorwheel[col:YG+col, 0]= 255 - np.floor(255*np.arange(0, YG, 1)/YG)
    colorwheel[col:YG+col, 1] = 255;
    col += YG;
    
    #GC
    colorwheel[col:GC+col, 1]= 255 
    colorwheel[col:GC+col, 2] = np.floor(255*np.arange(0, GC, 1)/GC)
    col += GC;
    
    #CB
    colorwheel[col:CB+col, 1]= 255 - np.floor(255*np.arange(0, CB, 1)/CB)
    colorwheel[col:CB+col, 2] = 255
    col += CB;
    
    #BM
    colorwheel[col:BM+col, 2]= 255 
    colorwheel[col:BM+col, 0] = np.floor(255*np.arange(0, BM, 1)/BM)
    col += BM;
    
    #MR
    colorwheel[col:MR+col, 2]= 255 - np.floor(255*np.arange(0, MR, 1)/MR)
    colorwheel[col:MR+col, 0] = 255

    return colorwheel

def compute_color(u, v):

    colorwheel = make_color_wheel()
    nan_u = np.isnan(u)
    nan_v = np.isnan(v)
    nan_u = np.where(nan_u)
    nan_v = np.where(nan_v) 
    
    u[nan_u] = 0
    u[nan_v] = 0
    v[nan_u] = 0 
    v[nan_v] = 0
    
    ncols = colorwheel.shape[0]
    radius = np.sqrt(u**2 + v**2)
    a = np.arctan2(-v, -u) / np.pi
    fk = (a+1) /2 * (ncols-1) # -1~1 maped to 1~ncols
    k0 = fk.astype(np.uint8)	 # 1, 2, ..., ncols
    k1 = k0+1;
    k1[k1 == ncols] = 0
    f = fk - k0
    
    img = np.empty([k1.shape[0], k1.shape[1],3])
    ncolors = colorwheel.shape[1]
    for i in range(ncolors):
        tmp = colorwheel[:,i]
        col0 = tmp[k0]/255
        col1 = tmp[k1]/255
        col = (1-f)*col0 + f*col1
        idx = radius <= 1
        col[idx] = 1 - radius[idx]*(1-col[idx]) # increase saturation with radius    
        col[~idx] *= 0.75 # out of range
        # img[:,:,2-i] = np.floor(255*col).astype(np.uint8)
        img[:,:,i] = np.floor(255*col).astype(np.uint8)

    return img.astype(np.uint8)

def compute_flow_image(flow):

    eps = sys.float_info.epsilon
    UNKNOWN_FLOW_THRESH = 1e9
    UNKNOWN_FLOW = 1e10
    
    u = flow[: , : , 0]
    v = flow[: , : , 1]
    
    maxu = -999
    maxv = -999
    
    minu = 999
    minv = 999
    
    maxrad = -1
    #fix unknown flow
    greater_u = np.where(u > UNKNOWN_FLOW_THRESH)
    greater_v = np.where(v > UNKNOWN_FLOW_THRESH)
    u[greater_u] = 0
    u[greater_v] = 0
    v[greater_u] = 0 
    v[greater_v] = 0
    
    maxu = max([maxu, np.amax(u)])
    minu = min([minu, np.amin(u)])
    
    maxv = max([maxv, np.amax(v)])
    minv = min([minv, np.amin(v)])
    rad = np.sqrt(np.multiply(u,u)+np.multiply(v,v)) 
    maxrad = max([maxrad, np.amax(rad)])
    # print('max flow: %.4f flow range: u = %.3f .. %.3f; v = %.3f .. %.3f\n' % (maxrad, minu, maxu, minv, maxv))
    
    u = u/(maxrad+eps)
    v = v/(maxrad+eps)
    img = compute_color(u, v)

    return img
