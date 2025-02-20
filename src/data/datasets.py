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
from pathlib import Path
from torchvision import transforms
from src.data.transforms import *
from src.data.masking_generator import TubeMaskingGenerator, RandomMaskingGenerator, CausalMaskingGenerator, CausalInterpolationMaskingGenerator, AutoregressiveMaskingGenereator
from src.data.co3d_dataset import Co3dLpDataset, AlignmentDataset, NerfDataset, MultiviewDataset
from src.data.vpt_dataset import PerspectiveDataset
from torch.utils.data import DataLoader
from src.util.metrics import create_label_index_map, create_label_index_map_imgnet
from torchvision import datasets, transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    
class DataAugmentationForVideoMAE(object):
    def __init__(self, args):
        self.input_mean = args.mean
        self.input_std = args.std
        normalize = GroupNormalize(self.input_mean, self.input_std)
        self.train_augmentation = GroupMultiScaleCrop(args.input_size, [1,.875, .75, .66])
        # self.train_augmentation = GroupMultiScaleCrop(args.input_size, [1])
        # self.color_augmentation = GroupColorJitter(contrast=(0.8, 1.2), saturation=(0.5, 1.5), hue=(-0.5, 0.5))
        # self.random_horizontal_flip = GroupRandomFlip()
        self.transform = transforms.Compose([                            
            self.train_augmentation,
            # self.random_horizontal_flip,
            # self.color_augmentation,
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
        is_binocular=False,
        reverse_sequence=False,
        length_divisor=args.data_length_divisor,
        camera_parameters_enabled=args.camera_params
    )
    print("Data Aug = %s" % str(transform))
    return dataset

def build_vpt_eval_loader(args):
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=args.mean, std=args.std)])

    train_dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'train_flip'), transform=transform)
    test_dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'test'), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'val'), transform=transform)
    human_dataset = PerspectiveDataset(Path(args.data_dir).parent, transforms=transform, split='human', task=args.task)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    human_loader = DataLoader(human_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, human_loader

def build_co3d_eval_loader(args, transform=None, return_all=False, convert2co3d=True):
    cifs = "/cifs/data/tserre_lrs/projects/prj_video_imagenet/"
    if not os.path.exists(cifs):
        cifs = "/cifs/data/tserre_lrs/projects/projects/prj_video_imagenet/"
    train_list_path = args.data_path
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
    if convert2co3d:
        label_dict = np.load(args.imgnet2co3d_label, allow_pickle=True).item()
    else:
        label_dict = None
        label_to_index_map = create_label_index_map_imgnet(args.data_root)

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
        co3d_dataloader_train = DataLoader(co3d_dataset_train, batch_size=args.eval_co3d_batch_size, shuffle=False, num_workers=args.num_workers)
        co3d_dataloader_val = DataLoader(co3d_dataset_val, batch_size=args.eval_co3d_batch_size, shuffle=False, num_workers=args.num_workers)
        return co3d_dataloader_train, co3d_dataloader_val, co3d_dataloader_test, imgnet_dataloader_test
    return co3d_dataloader_test, imgnet_dataloader_test
