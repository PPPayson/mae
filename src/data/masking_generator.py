import random
import torch
import numpy as np
import math
from einops import rearrange

def mask2attn_mask(mask, f2d=False, num_frames=1):
    if f2d:
        mask = rearrange(mask, 'b (t n) -> (b t) n', t=num_frames)
    cls_token_mask = torch.Tensor([True]).expand(mask.shape[0], -1).to(mask.device)
    mask = torch.cat((cls_token_mask, mask), dim=1).bool()
    return mask[:, None, :] * mask[:, :, None], mask

class NoMaskingGenerator:
    def __init__(self, input_size, mask_ratio=0):
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame =  self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame 
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)
        self.total_masks = self.frames * self.num_masks_per_frame
        print("num_mask_per_frame", self.num_masks_per_frame)
    def __repr__(self):
        repr_str = "Mask: total patches {}, mask patches {}".format(
            self.total_patches, self.total_masks
        )
        return repr_str

    def __call__(self):
        mask_per_frame = np.hstack([
            np.zeros(self.num_patches_per_frame),
        ])
        # np.random.shuffle(mask_per_frame)
        mask = np.tile(mask_per_frame, (self.frames,1)).flatten()
        # mask_per_frame = np.zeros((1, self.num_patches_per_frame))
        return mask     

class TubeMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame =  self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame 
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)
        self.total_masks = self.frames * self.num_masks_per_frame

    def __repr__(self):
        repr_str = "Mask: total patches {}, mask patches {}".format(
            self.total_patches, self.total_masks
        )
        return repr_str

    def __call__(self):
        mask_per_frame = np.hstack([
            np.zeros(self.num_patches_per_frame - self.num_masks_per_frame),
            np.ones(self.num_masks_per_frame),
        ])
        np.random.shuffle(mask_per_frame)
        mask = np.tile(mask_per_frame, (self.frames,1)).flatten()
        return mask 

class BlockMaskingGenerator:
    def __init__(self, input_size, mask_ratio, mask_ratio_var=0, pred_aspect_ratio=(0.3, 1/0.3)):
        self.frames, self.height, self.width = input_size
        self.mask_ratio = mask_ratio
        self.mask_ratio_var = mask_ratio_var
        self.num_patches_per_frame =  self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame 
        # self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)
        # self.total_masks = self.frames * self.num_masks_per_frame
        self.pred_start_epoch = -1
        self.epoch = 0
        self.log_aspect_ratio = tuple(map(lambda x: math.log(x), pred_aspect_ratio))
    def __repr__(self):
        repr_str = f'Variable Amount of Masks by Crop Size. Mask Ratio: {self.mask_ratio}.'
        return repr_str
    
    def get_pred_ratio(self):
        if hasattr(self, 'epoch') and self.epoch < self.pred_start_epoch:
            return 0
        if isinstance(self.mask_ratio, list):
            pred_ratio = []
            for prm, prv in zip(self.mask_ratio, self.mask_ratio_var):
                assert prm >= prv
                pr = random.uniform(prm - prv, prm + prv) if prv > 0 else prm
                pred_ratio.append(pr)
            pred_ratio = random.choice(pred_ratio)
        else:
            assert self.mask_ratio > self.mask_ratio_var
            pred_ratio = random.uniform(self.mask_ratio - self.mask_ratio_var, self.mask_ratio + \
                                    self.mask_ratio_var) if self.mask_ratio_var > 0 else self.mask_ratio

        return pred_ratio

    def __call__(self):
        # following BEiT (https://arxiv.org/abs/2106.08254), see at
        # https://github.com/microsoft/unilm/blob/b94ec76c36f02fb2b0bf0dcb0b8554a2185173cd/beit/masking_generator.py#L55
        height, width = self.height, self.width
        high = self.get_pred_ratio() * height * width
        mask = np.zeros((height, width), dtype=bool)
        mask_count = 0
        while mask_count < high:
            max_mask_patches = high - mask_count

            delta = 0
            for attempt in range(10):
                low = (min(height, width) // 3) ** 2 
                target_area = random.uniform(low, max_mask_patches)
                aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                if w < height and h < height:
                    top = random.randint(0, height - h)
                    left = random.randint(0, width - w)

                    num_masked = mask[top: top + h, left: left + w].sum()
                    if 0 < h * w - num_masked <= max_mask_patches:
                        for i in range(top, top + h):
                            for j in range(left, left + w):
                                if mask[i, j] == 0:
                                    mask[i, j] = 1
                                    delta += 1

                if delta > 0:
                    break

            if delta == 0:
                break
            else:
                mask_count += delta
        
        mask = np.tile(mask.flatten(), (self.frames,1)).flatten()
        return mask


class MultiCropBlockMaskingGenerator:
    def __init__(self, patch_size, mask_ratio, mask_ratio_var=0, pred_aspect_ratio=(0.3, 1/0.3), num_frames=8):
        self.frames = num_frames
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.mask_ratio_var = mask_ratio_var
        # self.num_patches = self.height * self.width # 14x14
        # self.num_mask = int(mask_ratio * self.num_patches)
        self.pred_start_epoch = -1
        self.epoch = 0
        self.log_aspect_ratio = tuple(map(lambda x: math.log(x), pred_aspect_ratio))
    def __repr__(self):
        repr_str = f'Variable Amount of Masks by Crop Size. Mask Ratio: {self.mask_ratio}.'
        return repr_str
    
    def get_pred_ratio(self):
        if hasattr(self, 'epoch') and self.epoch < self.pred_start_epoch:
            return 0
        
        if isinstance(self.mask_ratio, list):
            pred_ratio = []
            for prm, prv in zip(self.mask_ratio, self.mask_ratio_var):
                assert prm >= prv
                pr = random.uniform(prm - prv, prm + prv) if prv > 0 else prm
                pred_ratio.append(pr)
            pred_ratio = random.choice(pred_ratio)
        else:
            assert self.mask_ratio > self.mask_ratio_var
            pred_ratio = random.uniform(self.mask_ratio - self.mask_ratio_var, self.mask_ratio + \
                                    self.mask_ratio_var) if self.mask_ratio_var > 0 else self.mask_ratio

        return pred_ratio

    def __call__(self, imgs):
        masks = []
        for img in imgs:
            img = img[0]
            # following BEiT (https://arxiv.org/abs/2106.08254), see at
            # https://github.com/microsoft/unilm/blob/b94ec76c36f02fb2b0bf0dcb0b8554a2185173cd/beit/masking_generator.py#L55
            height, width = img.shape[1] // self.patch_size, img.shape[2] // self.patch_size
            high = self.get_pred_ratio() * height * width
            mask = np.zeros((height, width), dtype=bool)
            mask_count = 0
            while mask_count < high:
                max_mask_patches = high - mask_count

                delta = 0
                for attempt in range(10):
                    low = (min(height, width) // 3) ** 2 
                    target_area = random.uniform(low, max_mask_patches)
                    aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                    h = int(round(math.sqrt(target_area * aspect_ratio)))
                    w = int(round(math.sqrt(target_area / aspect_ratio)))
                    if w < height and h < height:
                        top = random.randint(0, height - h)
                        left = random.randint(0, width - w)

                        num_masked = mask[top: top + h, left: left + w].sum()
                        if 0 < h * w - num_masked <= max_mask_patches:
                            for i in range(top, top + h):
                                for j in range(left, left + w):
                                    if mask[i, j] == 0:
                                        mask[i, j] = 1
                                        delta += 1

                    if delta > 0:
                        break

                if delta == 0:
                    break
                else:
                    mask_count += delta
            mask = np.tile(mask.flatten(), (self.frames,1)).flatten()
            masks.append(mask)
        return masks

class RandomMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 3

        self.frames, self.height, self.width = input_size

        self.num_patches = self.height * self.width # 14x14
        self.num_mask = int(mask_ratio * self.num_patches)

    def __repr__(self):
        repr_str = "Mask: total patches {}, mask patches {}".format(
            self.num_patches * self.frames, self.num_mask * self.frames
        )
        return repr_str

    def __call__(self):
        masks = []
        for i in range(self.frames):
            mask = np.hstack([
                np.zeros(self.num_patches - self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
            masks.append(mask)
        return np.stack(masks) # [196*8]

class AutoregressiveMaskingGenereator:
    '''
    Masking all but first frame
    '''
    def __init__(self, input_size, mask_ratio):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 3
        self.frames, self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_mask = int(mask_ratio * self.num_patches)

    def __repr__(self):
        repr_str = "Mask: Causal with all but first frame masked"
        return repr_str

    def __call__(self):
        masks = []
        for i in range(self.frames):
            if i < 2:
                mask = np.zeros(self.num_patches)
            else:
                mask = np.ones(self.num_patches)
            masks.append(mask)
        return np.array(masks)

class CausalMaskingGenerator:
    '''
    Masking last frame
    '''
    def __init__(self, input_size, mask_ratio):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 3
        self.frames, self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_mask = int(mask_ratio * self.num_patches)

    def __repr__(self):
        repr_str = "Mask: causal with last frame masked"
        return repr_str

    def __call__(self):
        # masks = [0] * self.frames
        # masks[-1] = 1
        masks = []
        for i in range(self.frames):
            if i == (self.frames-1):
                mask = np.ones(self.num_patches)
            else:
                mask = np.zeros(self.num_patches)
            masks.append(mask)
        return np.array(masks)


class CausalInterpolationMaskingGenerator:
    '''
    Masking random middle frame
    '''
    def __init__(self, input_size, mask_ratio):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 3
        self.frames, self.height, self.width = input_size
        mask_ratio = float(1/self.frames) # Always going to mask one frame

        self.num_patches = self.height * self.width

    def __repr__(self):
        repr_str = "Mask: causal with a randomly masked middle frame"
        return repr_str

    def __call__(self):
        # First element is always visible
        masks = [np.zeros(self.num_patches)]

        # Uniform randomly pick the frame to mask
        index_to_mask = random.randint(1, self.frames-2)

        # Iterate over all frames except first and last
        for i in range(1, self.frames - 1):
            if i == index_to_mask:
                mask = np.ones(self.num_patches)
            else:
                mask = np.zeros(self.num_patches)
            masks.append(mask)

        masks.append(np.zeros(self.num_patches))
        # Don't add a mask to the last frame
        return np.array(masks)