import numpy as np
from pathlib import Path
import torch
import random
from torchvision.utils import make_grid
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import Compose, ToTensor
from timm.data.transforms import str_to_interp_mode
from itertools import chain
from scipy.spatial.transform import Rotation as R
from PIL import Image
import cv2
from scipy.ndimage.filters import gaussian_filter
import os

def save_as_overlay(img, mask, filename, percentile=99, save=True):

    vmax = np.max(mask)
    vmin = np.min(mask)
    mask = (mask-vmin)/(vmax-vmin)
    alpha = np.ones((mask.shape[0], mask.shape[1]))
    thresh = abs(np.percentile(mask, 90))
    alpha[np.logical_and(mask>=0, mask<thresh)] = 0
    alpha[np.logical_and(mask<0, mask>-thresh)] = 0
    mask = (mask+1)/2
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_VIRIDIS)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2RGBA)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
    overlay = cv2.addWeighted(heatmap, 0.6, img, 0.4, 0)
    img[alpha!=0] = overlay[alpha!=0]
    if save:
        cv2.imwrite(filename, img)
    return img

def create_label_index_map_imgnet(data_path):
    label_to_index_map = {}
    categories = sorted(os.listdir(os.path.join(data_path, 'train')))
    for i, c in enumerate(categories):
        label_to_index_map[c] = i
    return label_to_index_map

def create_label_index_map(anchor_file):
        '''
        Create a single label to index map to stay consistent across data splits
        '''
        label_to_index = {}
        index = 0
        with open(anchor_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                #label = line.split('/')[1]
                label = line.split()[0]
                if label not in label_to_index:
                    label_to_index[label] = index
                    index += 1
        return label_to_index

# Original implementation: https://github.com/pkmr06/pytorch-smoothgrad/blob/master/lib/gradients.py
def smooth_grad(model, x, stdev_spread=0.15, n_samples=25):
    x = x.cpu().numpy()
    stdev = stdev_spread * (np.max(x) - np.min(x))
    total_gradients = np.zeros((1, x.shape[-2], x.shape[-1]))
    for i in range(n_samples):
        noise = np.random.normal(0, stdev, x.squeeze().shape).astype(np.float32)
        x_plus_noise = x + noise
        x_plus_noise = torch.Tensor(x_plus_noise).cuda()
        x_plus_noise.requires_grad = True
        output = model(x_plus_noise)
        arg_idx = output.argmax()
        y = output[0][arg_idx]
        if x_plus_noise.grad is not None:
            x_plus_noise.grad.zero_()
        y.backward()
        grad = torch.amax(x_plus_noise.grad.abs(), dim=1).cpu().numpy()
        total_gradients += grad
    avg_gradients = total_gradients/n_samples

    return avg_gradients

# Original implementation: https://github.com/pkmr06/pytorch-smoothgrad/blob/master/lib/gradients.py
def batch_smooth_grad(model, x, stdev_spread=0.15, n_samples=25):
    stdev = stdev_spread * (torch.max(x) - torch.min(x))

    x = x.repeat(n_samples, 1, 1, 1)
    noise = np.random.normal(0, stdev.item(), x.shape).astype(np.float32)
    x_plus_noise = x + torch.Tensor(noise).to(x.device)
    x_plus_noise.requires_grad = True
    output = model(x_plus_noise)
    arg_idx = output.argmax(dim=-1)
    y = output[list(range(len(output))), arg_idx]


    if x_plus_noise.grad is not None:
        x_plus_noise.grad.zero_()
    y.backward(gradient=torch.ones_like(y))
    grad = torch.amax(x_plus_noise.grad.abs(), dim=1)
    avg_gradients = torch.mean(grad, dim=0)
    return avg_gradients


class MultiTaskLoss(torch.nn.Module):
  '''
    https://arxiv.org/abs/1705.07115
    https://github.com/ywatanabe1989/custom_losses_pytorch/blob/master/multi_task_loss.py
    usage
    is_regression = torch.Tensor([True, True, False]) # True: Regression/MeanSquaredErrorLoss, False: Classification/CrossEntropyLoss
    multitaskloss_instance = MultiTaskLoss(is_regression)

    params = list(model.parameters()) + list(multitaskloss_instance.parameters())
    torch.optim.Adam(params, lr=1e-3)

    model.train()
    multitaskloss.train()

    losses = torch.stack(loss0, loss1, loss3)
    multitaskloss = multitaskloss_instance(losses)
    '''
  def __init__(self, is_regression, reduction='none'):
    super(MultiTaskLoss, self).__init__()
    self.is_regression = is_regression
    self.n_tasks = len(is_regression)
    self.log_vars = torch.nn.Parameter(torch.zeros(self.n_tasks))
    self.reduction = reduction

  def forward(self, losses):
    dtype = losses.dtype
    device = losses.device
    stds = (torch.exp(self.log_vars)**(1/2)).to(device).to(dtype)
    self.is_regression = self.is_regression.to(device).to(dtype)
    coeffs = 1 / ( (self.is_regression+1)*(stds**2) )
    multi_task_losses = coeffs*losses + torch.log(stds)

    if self.reduction == 'sum':
      multi_task_losses = multi_task_losses.sum()
    if self.reduction == 'mean':
      multi_task_losses = multi_task_losses.mean()

    return multi_task_losses


