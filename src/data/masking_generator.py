import random
import numpy as np

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