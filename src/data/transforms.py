import math
import torch
import torchvision.transforms.functional as F
import warnings
import random
import numpy as np
import torchvision
from PIL import Image, ImageOps
import numbers
from collections.abc import Sequence
from typing import Optional

class GroupRandomFlip(object):
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, img_tuple):
        img_group, label = img_tuple
        out_images = []
        if torch.rand(1).item() < self.p:
            return (img_group, label)
        for img in img_group:
            out_images.append(F.hflip(img))
        return (out_images, label)


class GroupRandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img_tuple):
        img_group, label = img_tuple
        
        w, h = img_group[0].size
        th, tw = self.size

        out_images = list()

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        for img in img_group:
            assert(img.size[0] == w and img.size[1] == h)
            if w == tw and h == th:
                out_images.append(img)
            else:
                out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))

        return (out_images, label)


class GroupCenterCrop(object):
    def __init__(self, size):
        self.worker = torchvision.transforms.CenterCrop(size)

    def __call__(self, img_tuple):
        img_group, label = img_tuple
        return ([self.worker(img) for img in img_group], label)


class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).unsqueeze(1).unsqueeze(2)
        self.std = torch.tensor(std).unsqueeze(1).unsqueeze(2)

    def __call__(self, tensor_tuple):
        # import pdb; pdb.set_trace()
        tensor, label = tensor_tuple
        #rep_mean = self.mean * (tensor.size()[0]//len(self.mean))
        #rep_std = self.std * (tensor.size()[0]//len(self.std))
        
        # TODO: make efficient
        # for t, m, s in zip(tensor, rep_mean, rep_std):
        #     t.sub_(m).div_(s)
        tensor.sub_(self.mean).div_(self.std)
        return (tensor,label)

    
class GroupScale(object):
    """ Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.worker = torchvision.transforms.Resize(size, interpolation)

    def __call__(self, img_tuple):
        img_group, label = img_tuple
        return ([self.worker(img) for img in img_group], label)


class GroupColorJitter(object):

    def __init__(self, brightness=0, contrast=(0.2, 1.2), saturation=(0.5, 1.5), hue=(-0.5, 0.5), shared=True):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)
        self.shared=shared
   
    @torch.jit.unused
    def _check_input(self, value, name, center=1, bound=(0, float("inf")), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(f"If {name} is a single number, it must be non negative.")
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            value = [float(value[0]), float(value[1])]
        else:
            raise TypeError(f"{name} should be a single number or a list/tuple with length 2.")

        if not bound[0] <= value[0] <= value[1] <= bound[1]:
            raise ValueError(f"{name} values should be between {bound}, but got {value}.")

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            return None
        else:
            return tuple(value)

    def __call__(self, img_tuple):
        img_group, label = img_tuple
        if self.shared:
            fn_idx, brightness_factor, \
            contrast_factor, saturation_factor, \
            hue_factor = torchvision.transforms.ColorJitter.get_params(
                        self.brightness, self.contrast, self.saturation, self.hue
                        )
            for fn_id in fn_idx:
                if fn_id == 0 and brightness_factor is not None:
                    img_group = [F.adjust_brightness(img, brightness_factor) for img in img_group]
                elif fn_id == 1 and contrast_factor is not None:
                    img_group = [F.adjust_contrast(img, contrast_factor) for img in img_group]
                elif fn_id == 2 and saturation_factor is not None:
                    img_group = [F.adjust_saturation(img,saturation_factor) for img in img_group]
                elif fn_id == 3 and hue_factor is not None:
                    img_group = [F.adjust_hue(img,hue_factor) for img in img_group]
            return (img_group, label)

        else:
            transformed = []
            for img in img_group:
                fn_idx, brightness_factor, \
                contrast_factor, saturation_factor, \
                hue_factor = torchvision.transforms.ColorJitter.get_params(
                            self.brightness, self.contrast, self.saturation, self.hue
                            )
                for fn_id in fn_idx:
                    if fn_id == 0 and brightness_factor is not None:
                        img = F.adjust_brightness(img, brightness_factor)
                    elif fn_id == 1 and contrast_factor is not None:
                        img = F.adjust_contrast(img, contrast_factor)
                    elif fn_id == 2 and saturation_factor is not None:
                        img = F.adjust_saturation(img, saturation_factor)
                    elif fn_id == 3 and hue_factor is not None:
                        img = F.adjust_hue(img, hue_factor)
                transformed.append(img)
            return (transformed, label)

class GroupRandomGrayScale(object):
    def __init__(self, p=0.1, shared=True):
        self.p = p
        self.shared=shared
    def __call__(self, img_tuple):
        img_group, label = img_tuple
        transformed = []
        num_output_channels, _, _ = F.get_dimensions(img_group[0])
        if not self.shared:
            for img in img_group:
                if torch.rand(1).item() < self.p:
                    transformed.append(F.rgb_to_grayscale(img, num_output_channels=num_output_channels))
                else:
                    transformed.append(img)
        else:
            if torch.rand(1).item() < self.p:
                for img in img_group:
                    transformed.append(F.rgb_to_grayscale(img, num_output_channels=num_output_channels))
            else:
                transformed = img_group
        return (transformed, label)

class GroupGaussianBlur(object):
    def __init__(self, sigma=(0.1, 2.0), shared=True):
        # self.kernel_size = torchvision.transforms.transforms._setup_size(kernel_size, "Kernel size should be a tuple/list of two integers")
        # for ks in self.kernel_size:
        #     if ks <= 0 or ks % 2 == 0:
        #         raise ValueError("Kernel size value should be an odd and positive number.")

        if isinstance(sigma, numbers.Number):
            if sigma <= 0:
                raise ValueError("If sigma is a single number, it must be positive.")
            sigma = (sigma, sigma)
        elif isinstance(sigma, Sequence) and len(sigma) == 2:
            if not 0.0 < sigma[0] <= sigma[1]:
                raise ValueError("sigma values should be positive and of the form (min, max).")
        else:
            raise ValueError("sigma should be a single number or a list/tuple with length 2.")

        self.sigma = sigma
        self.shared = shared
    
    def __call__(self, img_tuple):
        img_group, label = img_tuple
        if self.shared:
            sigma = torchvision.transforms.GaussianBlur.get_params(self.sigma[0], self.sigma[1])
            kernel_size = (math.ceil(6*sigma[0])+1, math.ceil(6*sigma[1])+1)
            img_group = [F.gaussian_blur(img, kernel_size, [sigma, sigma]) for img in img_group]
            return (img_group, label)
        else:
            transformed = []
            for img in img_group:
                sigma = torchvision.transforms.GaussianBlur.get_params(self.sigma[0], self.sigma[1])
                kernel_size = (2*math.ceil(3*sigma)+1, 2*math.ceil(3*sigma)+1)
                transformed.append(F.gaussian_blur(img, kernel_size, [sigma, sigma]))
            return (transformed, label)

class GroupRandomSolarize(object):
    def __init__(self, threshold, p=0.5, shared=True):
        self.threshold = threshold
        self.p = p
        self.shared = shared
    
    def __call__(self, img_tuple):
        img_group, label = img_tuple
        if self.shared:
            if torch.rand(1).item() < self.p:
                img_group = [F.solarize(img, self.threshold) for img in img_group]
            return (img_group, label)
        else:
            transformed = []
            for img in img_group:
                if torch.rand(1).item() < self.p:
                    img = F.solarize(img, self.threshold)
                transformed.append(img)
            return (transformed, label)

class GroupRandomResizedCrop(object):
    def __init__(self,
                input_size,
                scale=(0.08, 1.0),
                ratio=(3.0 / 4.0, 4.0 / 3.0),
                interpolation=F.InterpolationMode.BILINEAR,
                antialias: Optional[bool]=True,
                shared: bool = True):
        super().__init__()
        self.size = torchvision.transforms.transforms._setup_size(input_size, error_msg="Please provide only two dimensions (h, w) for size.")
        if not isinstance(scale, Sequence):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, Sequence):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")

        if isinstance(interpolation, int):
            interpolation = torchvision.transforms.transforms._interpolation_modes_from_int(interpolation)

        self.interpolation = interpolation
        self.antialias = antialias
        self.scale = scale
        self.ratio = ratio
        self.shared = shared

    def __call__(self, img_tuple):
        img_group, label = img_tuple
        if self.shared:
            # Assume all images are of the same size
            img = img_group[0]
            i, j, h, w = torchvision.transforms.transforms.RandomResizedCrop.get_params(img, self.scale, self.ratio)
            for i, img in enumerate(img_group):
                img_group[i] = F.resized_crop(img, i, j, h, w, self.size, self.interpolation, antialias=self.antialias)
        else:
            for i, img in enumerate(img_group):
                i, j, h, w = torchvision.transforms.transforms.RandomResizedCrop.get_params(img, self.scale, self.ratio)
                img_group[i] = F.resized_crop(img, i, j, h, w, self.size, self.interpolation, antialias=self.antialias)
        return (img_group, label)

class GroupMultiScaleCrop(object):

    def __init__(self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True):
        self.scales = scales if scales is not None else [1, .875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]
        # self.interpolation = Image.BILINEAR
        self.interpolation =F.InterpolationMode.BILINEAR

    def __call__(self, img_tuple):
        img_group, label = img_tuple
        im_size = img_group[0].size()[1:]

        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        
        crop_img_group = [F.crop(img, offset_w, offset_h, crop_w, crop_h) for img in img_group]
        ret_img_group = [F.resize(img, (self.input_size[0], self.input_size[1]), self.interpolation) for img in crop_img_group]

        return (ret_img_group, label)

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]

        # find a crop size
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [self.input_size[1] if abs(x - self.input_size[1]) < 3 else x for x in crop_sizes]
        crop_w = [self.input_size[0] if abs(x - self.input_size[0]) < 3 else x for x in crop_sizes]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(image_w, image_h, crop_pair[0], crop_pair[1])

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(self.more_fix_crop, image_w, image_h, crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = list()
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center

        if more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter
        return ret


class Stack(object):

    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_tuple):
        img_group, label = img_tuple
        
        if isinstance(img_group[0], torch.Tensor):
            return (torch.stack(img_group), label)
        elif img_group[0].mode == 'L':
            return (np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2), label)
        elif img_group[0].mode == 'RGB':
            if self.roll:
                return (np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2), label)
            else:
                return (np.concatenate(img_group, axis=2), label)        


class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic_tuple):
        pic, label = pic_tuple
        if isinstance(pic, torch.Tensor):
            img = pic
        elif isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            # img = img.transpose(0, 1).transpose(0, 2).contiguous()
            img = img.permute((2, 0, 1)).contiguous()
        return (img.float().div(255.) if self.div else img.float(), label)


class IdentityTransform(object):

    def __call__(self, data):
        return data

