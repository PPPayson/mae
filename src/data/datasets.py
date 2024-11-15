# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import json
import PIL
from torchvision import transforms
from src.data.transforms import *
from src.data.masking_generator import TubeMaskingGenerator, RandomMaskingGenerator, CausalMaskingGenerator, CausalInterpolationMaskingGenerator, AutoregressiveMaskingGenereator
from src.data.co3d_dataset import Co3dLpDataset, AlignmentDataset, NerfDataset, MultiviewDataset
from torch.utils.data import DataLoader
from src.util.metrics import create_label_index_map
from torchvision import datasets, transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

class DataAugmentationForVideoMAE(object):
    def __init__(self, args):

        # self.input_mean = [0.485, 0.456, 0.406]  # IMAGENET1k_DEFAULT_MEAN
        # self.input_std = [0.229, 0.224, 0.225]  # IMAGENET1k_DEFAULT_STD
        #if 'beit' in args.model:
        # self.input_mean = [0.5, 0.5, 0.5]  # IMAGENET21k_DEFAULT_MEAN
        # self.input_std = [0.5, 0.5, 0.5]  # IMAGENET21k_DEFAULT_STD
        self.input_mean = args.mean
        self.input_std = args.std
        normalize = GroupNormalize(self.input_mean, self.input_std)
        self.train_augmentation = GroupMultiScaleCrop(args.input_size, [1,.875, .75, .66])
        self.color_augmentation = GroupColorJitter(contrast=(0.8, 1.2), saturation=(0.5, 1.5), hue=(-0.5, 0.5))
        self.transform = transforms.Compose([                            
            self.train_augmentation,
            self.color_augmentation,
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            normalize,
        ])
        if args.mask_type == 'tube':
            self.masked_position_generator = TubeMaskingGenerator(
                args.window_size, args.mask_ratio
            )
        elif args.mask_type == 'causal':
            args.mask_ratio = 1/(args.num_frames) #Mask the last frame
            self.masked_position_generator = CausalMaskingGenerator(args.window_size, args.mask_ratio)
        elif args.mask_type == "autoregressive":
            args.mask_ratio = (args.num_frames-1)/args.num_frames
            self.masked_position_generator = AutoregressiveMaskingGenereator(args.window_size, args.mask_ratio)
        elif args.mask_type == 'causal_interpol':
            self.masked_position_generator = CausalInterpolationMaskingGenerator(args.window_size, args.mask_ratio)
        else:
            self.masked_position_generator = RandomMaskingGenerator(
                args.window_size, args.mask_ratio
            )

    def __call__(self, images):
        process_data, _ = self.transform(images)
        return process_data, self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForVideoMAE,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr

def build_frames_dataset(args, transform=None, is_train=True):
    if transform is None:
        transform = DataAugmentationForVideoMAE(args)
        mae_transform = True
    else:
        mae_transform = False
    if is_train:
        datapath = args.data_path
    else:
        datapath = args.data_val_path

    dataset = NerfDataset(data_root=args.data_root, data_list=datapath, transform=transform, mae_transform=mae_transform)
    return dataset

def build_pretraining_dataset(args, is_train=True):
    transform = DataAugmentationForVideoMAE(args)
    if is_train:
        datapath = args.data_path
    else:
        datapath = args.data_val_path

    dataset = MultiviewDataset(
        data_root = args.data_root,
        data_list=datapath,
        new_length=args.num_frames,
        new_step=args.sampling_rate,
        transform=transform,
        is_binocular=args.binocular,
        reverse_sequence=args.reverse_sequence,
        length_divisor=args.data_length_divisor,
        camera_parameters_enabled=args.camera_params
    )
    print("Data Aug = %s" % str(transform))
    return dataset

def build_co3d_eval_loader(args, transform=None, return_all=False):
    cifs = "/cifs/data/tserre_lrs/projects/prj_video_imagenet/"
    if not os.path.exists(cifs):
        cifs = "/cifs/data/tserre_lrs/projects/projects/prj_video_imagenet/"
    train_list_path = args.data_list
    data_root = os.path.join(cifs, 'PeRFception/data/co3d_v2/binocular_trajectory/')
    data_path = os.path.join(cifs, 'Evaluation/')
    test_data_path = args.clickmaps_path
    test_human_results = args.clickmaps_human_path

    test_imgnet_path = args.imgnet_clickmaps_path
    test_imgnet_human_resutls = args.imgnet_clickmaps_human_path
    label_to_index_map = create_label_index_map(train_list_path)
    
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=args.mean, std=args.std)])


    co3d_dataset_test = AlignmentDataset(numpy_file=test_data_path, human_results_file=test_human_results, label_to_index=label_to_index_map, transform=transform)
    co3d_dataloader_test = DataLoader(co3d_dataset_test, batch_size=1, num_workers=args.num_workers, shuffle=False)
    
    label_dict = np.load(args.imgnet2co3d_label, allow_pickle=True).item()
    imgnet_dataset_test = AlignmentDataset(numpy_file=test_imgnet_path, human_results_file=test_imgnet_human_resutls,
                                             label_to_index=label_to_index_map, label_dict=label_dict, transform=transform, dataset_name='imgnet')
    imgnet_dataloader_test = DataLoader(imgnet_dataset_test, batch_size=1, num_workers=args.num_workers, shuffle=False)

    if return_all:
        co3d_dataset_train = Co3dLpDataset(root=data_root,
                train = True,
                transform=transform,
                datapath = os.path.join(data_path,"filtered_co3d_train.txt"),
                train_start_frame=0,
                train_end_frame=40,
                val_start_frame=41, 
                val_end_frame=49)
        co3d_dataset_val = Co3dLpDataset(root=data_root,
                train = False,
                transform=transform,
                datapath = os.path.join(data_path, "filtered_co3d_test.txt"),
                train_start_frame=0,
                train_end_frame=40,
                val_start_frame=41, 
                val_end_frame=49)
        co3d_dataloader_train = DataLoader(co3d_dataset_train, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        co3d_dataloader_val = DataLoader(co3d_dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        return co3d_dataloader_train, co3d_dataloader_val, co3d_dataloader_test, imgnet_dataloader_test
    return co3d_dataloader_test, imgnet_dataloader_test


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

