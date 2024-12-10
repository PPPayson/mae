import re
import os
import sys
import torch
import yaml
import numpy as np
import pandas as pd
from torch.nn import functional as F
from scipy.stats import spearmanr
from tqdm import tqdm
from torchvision.transforms import functional as tvF


def circle_kernel(size, sigma=None):
    """
    Create a flat circular kernel where the values are the average of the total number of on pixels in the filter.

    Args:
        size (int): The diameter of the circle and the size of the kernel (size x size).
        sigma (float, optional): Not used for flat kernel. Included for compatibility. Default is None.

    Returns:
        torch.Tensor: A 2D circular kernel normalized so that the sum of its elements is 1.
    """
    # Create a grid of (x,y) coordinates
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
    center = (size - 1) / 2
    radius = (size - 1) / 2

    # Create a mask for the circle
    mask = (x - center) ** 2 + (y - center) ** 2 <= radius ** 2

    # Initialize kernel with zeros and set ones inside the circle
    kernel = torch.zeros((size, size), dtype=torch.float32)
    kernel[mask] = 1.0

    # Normalize the kernel so that the sum of all elements is 1
    num_on_pixels = mask.sum()
    if num_on_pixels > 0:
        kernel = kernel / num_on_pixels

    # Add batch and channel dimensions
    kernel = kernel.unsqueeze(0).unsqueeze(0)

    return kernel

def compute_spearman_correlation(map1, map2):
    """
    Compute the Spearman correlation between two maps.

    Args:
        map1 (np.ndarray): The first map.
        map2 (np.ndarray): The second map.

    Returns:
        float: The Spearman correlation coefficient, or NaN if computation is not possible.
    """
    filtered_map1 = map1.flatten()
    filtered_map2 = map2.flatten()

    if filtered_map1.size > 1 and filtered_map2.size > 1:
        correlation, _ = spearmanr(filtered_map1, filtered_map2)
        return correlation
    else:
        return float('nan')

def fast_ious(v1, v2):
    """
    Compute the IoU between two images.

    Args:
        image_1 (np.ndarray): The first image.
        image_2 (np.ndarray): The second image.

    Returns:
        float: The IoU between the two images.
    """
    # Compute intersection and union
    intersection = np.logical_and(v1, v2).sum()
    union = np.logical_or(v1, v2).sum()

    # Compute IoU
    iou = intersection / union if union != 0 else 0.0

    return iou
    
def gaussian_kernel(size, sigma):
    """
    Create a Gaussian kernel.

    Args:
        size (int): Size of the kernel.
        sigma (float): Standard deviation of the Gaussian distribution.

    Returns:
        torch.Tensor: A 2D Gaussian kernel with added batch and channel dimensions.
    """
    x_range = torch.arange(-(size-1)//2, (size-1)//2 + 1, 1)
    y_range = torch.arange((size-1)//2, -(size-1)//2 - 1, -1)

    xs, ys = torch.meshgrid(x_range, y_range, indexing='ij')
    kernel = torch.exp(-(xs**2 + ys**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)
    
    kernel = kernel / kernel.sum()
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    
    return kernel

def convolve(heatmap, kernel, double_conv=False):
    """
    Apply Gaussian blur to a heatmap.

    Args:
        heatmap (torch.Tensor): The input heatmap (3D or 4D tensor).
        kernel (torch.Tensor): The Gaussian kernel.

    Returns:
        torch.Tensor: The blurred heatmap (3D tensor).
    """
    # heatmap = heatmap.unsqueeze(0) if heatmap.dim() == 3 else heatmap
    blurred_heatmap = F.conv2d(heatmap, kernel, padding='same')
    if double_conv:
        blurred_heatmap = F.conv2d(blurred_heatmap, kernel, padding='same')
    return blurred_heatmap  # [0]

def integrate_surface(iou_scores, x, z, average_areas=True, normalize=False):
    # Integrate along x axis (classifier thresholds)
    if len(z) == 1:
        return iou_scores.mean()

    int_x = np.trapz(iou_scores, x, axis=1)
    
    if average_areas:
        return int_x.mean()

    # Integrate along z axis (label thresholds)
    int_xz = np.trapz(int_x, z)

    if normalize:
        x_range = x[-1] - x[0]
        z_range = z[-1] - z[0]
        total_area = x_range * z_range
        int_xz /= total_area

    return int_xz

def compute_RSA(map1, map2):
    """
    Compute the RSA between two maps.

    Returns:    
        float: The RSA between the two maps.
    """
    import pdb; pdb.set_trace()
    return np.corrcoef(map1, map2)[0, 1]

def compute_AUC(
        pred_map,
        target_map,
        prediction_threshs=21,
        target_threshold_min=0.25,
        target_threshold_max=0.75,
        target_threshs=9):
    """
    We will compute IOU between pred and target over multiple threshodls of the target map.

    Args:
        map1 (np.ndarray): The first map.
        map2 (np.ndarray): The second map.  

    Returns:
        float: The AUC between the two maps.
    """
    # Make sure both maps are probability distributions
    # if normalize:
    #     map1 = map1 / map1.sum()
    #     map2 = map2 / map2.sum()
    inner_thresholds = np.linspace(0, 1, prediction_threshs)
    target_thresholds = np.linspace(target_threshold_min, target_threshold_max, target_threshs)
    # thresholds = [0.25, 0.5, 0.75, 1]
    thresh_ious = []
    for outer_t in target_thresholds:
        thresh_target_map = (target_map >= outer_t).astype(int).ravel()
        ious = []
        for t in inner_thresholds:
            thresh_pred_map = (pred_map >= t).astype(int).ravel()
            iou = fast_ious(thresh_target_map, thresh_pred_map)
            ious.append(iou)
        thresh_ious.append(np.asarray(ious))
    thresh_ious = np.stack(thresh_ious, 0)
    return integrate_surface(thresh_ious, inner_thresholds, target_thresholds, normalize=True)

def compute_crossentropy(map1, map2):
    """
    Compute the cross-entropy between two maps.

    Args:
        map1 (np.ndarray): The first map.
        map2 (np.ndarray): The second map.  

    Returns:
        float: The cross-entropy between the two maps.
    """
    map1 = torch.from_numpy(map1).float().ravel()
    map2 = torch.from_numpy(map2).float().ravel()
    return F.cross_entropy(map1, map2).numpy()

def alt_gaussian_kernel(size=10, sigma=10):
    """
    Generates a 2D Gaussian kernel.

    Parameters
    ----------
    size : int, optional
        Kernel size, by default 10
    sigma : int, optional
        Kernel sigma, by default 10

    Returns
    -------
    kernel : torch.Tensor
        A Gaussian kernel.
    """
    x_range = torch.arange(-(size-1)//2, (size-1)//2 + 1, 1)
    y_range = torch.arange((size-1)//2, -(size-1)//2 - 1, -1)

    xs, ys = torch.meshgrid(x_range, y_range, indexing='ij')
    kernel = torch.exp(-(xs**2 + ys**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)
    
    kernel = kernel / kernel.sum()
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    
    return kernel

def alt_gaussian_blur(heatmap, kernel):
    """
    Blurs a heatmap with a Gaussian kernel.

    Parameters
    ----------
    heatmap : torch.Tensor
        The heatmap to blur.
    kernel : torch.Tensor
        The Gaussian kernel.

    Returns
    -------
    blurred_heatmap : torch.Tensor
        The blurred heatmap.
    """
    # Ensure heatmap and kernel have the correct dimensions
    heatmap = heatmap.unsqueeze(0) if heatmap.dim() == 3 else heatmap
    blurred_heatmap = torch.nn.functional.conv2d(heatmap, kernel, padding='same')

    return blurred_heatmap[0]
    
def gaussian_blur(heatmap, kernel):
    """
    Blurs a heatmap with a Gaussian kernel.

    Parameters
    ----------
    heatmap : torch.Tensor
        The heatmap to blur.
    kernel : torch.Tensor
        The Gaussian kernel.

    Returns
    -------
    blurred_heatmap : torch.Tensor
        The blurred heatmap.
    """
    # Ensure heatmap and kernel have the correct dimensions
    heatmap = heatmap.unsqueeze(0) if heatmap.dim() == 3 else heatmap
    blurred_heatmap = torch.nn.functional.conv2d(heatmap, kernel, padding='same')

    return blurred_heatmap[0]
