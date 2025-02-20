import os
import glob
import torch
import numpy as np
from scipy.stats import spearmanr
from torchvision.transforms import functional as tvF


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


def sliding_window_coorrelation(img_1, img_2, window_size=32, stride=None):
    assert img_1.shape == img_2.shape
    if stride is None:
        stride = window_size//2
    h, w = img_1.shape
    correlations = []
    # Replace with torch unfold
    for i in range(int((h-window_size)/stride)+1):
        for j in range(int((w-window_size)/stride)+1):
            window_1 = img_1[i*stride:(i*stride)+window_size, j*stride:(j*stride)+window_size]
            window_2 = img_2[i*stride:(i*stride)+window_size, j*stride:(j*stride)+window_size]
            correlations.append(compute_spearman_correlation(window_1, window_2))
    return np.nanmean(correlations)

if __name__ == "__main__":
    asset_path = 'outs/collected_maps'
    human_maps_path = os.path.join(asset_path, 'human')
    model_maps_path = os.path.join(asset_path, 'eva_giant_patch14_224.clip_ft_in1k')
    human_maps = sorted(glob.glob(os.path.join(human_maps_path, '*.npy')))
    model_maps = sorted(glob.glob(os.path.join(model_maps_path, '*.npy')))
    window_size = 16
    for index, h in enumerate(human_maps):
        human_map = torch.Tensor(np.load(h))[None, :, : ,:]
        human_map = tvF.resize(human_map, 256)
        human_map = tvF.center_crop(human_map, (224, 224))
        full_human_maps = human_map.clone().squeeze().numpy()
        human_map = human_map.mean(1).squeeze().numpy()
        human_map = (human_map - human_map.min()) / (human_map.max() - human_map.min())

        model_map = np.load(model_maps[index])
        model_map = (model_map - model_map.min()) / (model_map.max() - model_map.min())
        full_score = compute_spearman_correlation(model_map, human_map)
        score = sliding_window_coorrelation(human_map, model_map, window_size=window_size)
        human_ceiling = []
        full_human_ceiling = []
        for _ in range(50):
            n = full_human_maps.shape[0]
            rand_perm = np.random.permutation(n)
            fh = rand_perm[:(n // 2)]
            sh = rand_perm[(n // 2):]
            fh = full_human_maps[fh].mean(0)
            sh = full_human_maps[sh].mean(0)
            fh = (fh - fh.min()) / (fh.max() - fh.min())
            sh = (sh - sh.min()) / (sh.max() - sh.min())
            full_human_ceiling.append(compute_spearman_correlation(fh, sh))
            human_ceiling.append(sliding_window_coorrelation(fh, sh, window_size=window_size))
        human_ceiling = np.mean(human_ceiling)
        full_human_ceiling = np.mean(full_human_ceiling)
        print(h.split('/')[-1])
        print('ceiling', human_ceiling)
        print('score', score)
        # print('normalized score', score/human_ceiling)
        # print('full_score', full_score)
        # print('full_ceiling', full_human_ceiling)