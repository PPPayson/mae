import os
import cv2
import json
import random
import numpy as np
from numpy.lib.function_base import disp
import torch
from PIL import Image
from torchvision import transforms
import warnings
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
from sklearn import preprocessing
from src.util.misc import matrix_to_quaternion, camera_to_category
from turbojpeg import TurboJPEG
from einops import rearrange
jpegturbo = TurboJPEG()

def spatial_sampling(
    frames,
    spatial_idx=-1,
    min_scale=256,
    max_scale=320,
    crop_size=224,
    random_horizontal_flip=True,
    inverse_uniform_sampling=False,
    aspect_ratio=None,
    scale=None,
    motion_shift=False,
):
    """
    Perform spatial sampling on the given video frames. If spatial_idx is
    -1, perform random scale, random crop, and random flip on the given
    frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
    with the given spatial_idx.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `num frames` x `height` x `width` x `channel`.
        spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
            or 2, perform left, center, right crop if width is larger than
            height, and perform top, center, buttom crop if height is larger
            than width.
        min_scale (int): the minimal size of scaling.
        max_scale (int): the maximal size of scaling.
        crop_size (int): the size of height and width used to crop the
            frames.
        inverse_uniform_sampling (bool): if True, sample uniformly in
            [1 / max_scale, 1 / min_scale] and take a reciprocal to get the
            scale. If False, take a uniform sample from [min_scale,
            max_scale].
        aspect_ratio (list): Aspect ratio range for resizing.
        scale (list): Scale range for resizing.
        motion_shift (bool): Whether to apply motion shift for resizing.
    Returns:
        frames (tensor): spatially sampled frames.
    """
    assert spatial_idx in [-1, 0, 1, 2]
    if spatial_idx == -1:
        if aspect_ratio is None and scale is None:
            frames, _ = video_transforms.random_short_side_scale_jitter(
                images=frames,
                min_size=min_scale,
                max_size=max_scale,
                inverse_uniform_sampling=inverse_uniform_sampling,
            )
            frames, _ = video_transforms.random_crop(frames, crop_size)
        else:
            transform_func = (
                video_transforms.random_resized_crop_with_shift
                if motion_shift
                else video_transforms.random_resized_crop
            )
            frames = transform_func(
                images=frames,
                target_height=crop_size,
                target_width=crop_size,
                scale=scale,
                ratio=aspect_ratio,
            )
        if random_horizontal_flip:
            frames, _ = video_transforms.horizontal_flip(0.5, frames)
    else:
        # The testing is deterministic and no jitter should be performed.
        # min_scale, max_scale, and crop_size are expect to be the same.
        assert len({min_scale, max_scale, crop_size}) == 1
        frames, _ = video_transforms.random_short_side_scale_jitter(
            frames, min_scale, max_scale
        )
        frames, _ = video_transforms.uniform_crop(frames, crop_size, spatial_idx)
    return frames


def tensor_normalize(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor

class MultiviewDataset(torch.utils.data.Dataset):
    """Load your own video classification dataset.
    Parameters
    ----------
    root : str, required.
        Path to the root folder storing the dataset.
    setting : str, required.
        A text file describing the dataset, each line per video sample.
        There are three items in each line: (1) video path; (2) video length and (3) video label.
    new_length : int, default 1.
        The length of returned video clip. Default is a single image, but it can be multiple video frames.
        For example, new_length=16 means we will extract a video clip of consecutive 16 frames.
    new_step : int, default 1.
        Temporal sampling rate. For example, new_step=1 means we will extract a video clip of consecutive frames.
        new_step=2 means we will extract a video clip of every other frame.
    transform : function, default None.
        A function that takes data and label and transforms them.
    """
    def __init__(self,
                 data_root,
                 data_list,
                 new_length=4,
                 new_step=1,
                 transform=None,
                 is_binocular=False,
                 reverse_sequence=False,
                 length_divisor=1,
                 camera_parameters_enabled=False
                 ):

        super().__init__()

        self.data_root = data_root
        self.data_list = data_list

        self.new_length = new_length
        self.new_step = new_step

        self.transform = transform
        self.is_binocular = is_binocular
        self.reverse_sequence = reverse_sequence
        self.length_divisor = length_divisor

        self.camera_parameters_enabled = camera_parameters_enabled
        self.clips = self.make_dataset_samples()
        
        num_valid_starts_per_clip = [max(0, (len(clip[2:]) - (self.new_length * self.new_step) + 1)) for clip in self.clips]
        self.num_frames_per_clip = [len(clip[2:]) for clip in self.clips]
        self.length = sum(num_valid_starts_per_clip)

        # Might need to add check for first video has no valid start
        self.start_indices = [[0, 0]]*num_valid_starts_per_clip[0]
        for i in range(len(num_valid_starts_per_clip)-1):
            self.start_indices += [[i+1, self.start_indices[-1][1]+num_valid_starts_per_clip[i]]] * num_valid_starts_per_clip[i+1]
        self.encoder_logits = None
        self.encoder_labels = None
        # self.encoder_names = None
        labels = []
        for clip in self.clips:
            labels.append(clip[0])

        labels = np.unique(labels)
        self.label_preprocessing = preprocessing.LabelEncoder()
        self.label_preprocessing.fit(np.array(labels))
        if len(self.clips) == 0:
            raise(RuntimeError("Found 0 video clips in subfolders of: " + data_root + "\n"
                               "Check your data directory (opt.data-dir)."))

    def set_logits(self, encoder_logits, encoder_labels):
        self.encoder_logits = []
        cur_idx = 0
        for i in range(len(self.num_frames_per_clip)):
            self.encoder_logits.append(encoder_logits[cur_idx:cur_idx+self.num_frames_per_clip[i]])
            cur_idx += self.num_frames_per_clip[i]
    def __getitem__(self, index):
        #clip_idx = index // self.num_valid_starts_per_clip[index]
        #start_frame_idx = index % self.num_valid_starts_per_clip[index]
        clip_idx = self.start_indices[index][0]
        start_frame_idx = index - self.start_indices[index][1]
        clip = self.clips[clip_idx]
        label = clip[0]
        label = self.label_preprocessing.transform([label])
        label = torch.tensor(label)[0].repeat(self.new_length)

        if self.camera_parameters_enabled:
            images, camera_params, camera_cats, features = self.load_frames_and_params(clip, start_frame_idx, clip_idx)
            camera_cats = torch.Tensor(camera_cats).to(torch.long)
            camera_params = torch.vstack(camera_params)
            process_data, mask = self.transform((images, None)) # T,C,H,W
            if self.is_binocular:
                mask = np.tile(mask, (1, 2, 1))

            # print("MASK IS:", mask)

            process_data = process_data.view((self.new_length, 3) + process_data.size()[-2:]).transpose(0,1)  # T*C,H,W -> T,C,H,W -> C,T,H,W
            if self.encoder_logits != None:
                features = torch.vstack(features)
                return (process_data, mask, camera_params, camera_cats, features, label)
            return (process_data, mask, camera_params, camera_cats, label)

        else:
            images, features  = self.load_frames(clip, start_frame_idx, clip_idx)
            process_data, mask = self.transform((images, None)) # T,C,H,W
            if self.is_binocular:
                mask = np.tile(mask, (1, 2, 1))

            process_data = process_data.view((self.new_length, 3) + process_data.size()[-2:]).transpose(0,1)  # T*C,H,W -> T,C,H,W -> C,T,H,W
            if self.encoder_logits is not None:
                features = torch.vstack(features)
                return (process_data, mask, features, label)
            return (process_data, mask, label)

    def __len__(self):
        # Multiply by the number of frames in each sequence
        # For Pre-training Co3D, multiply by 50
        # For Greebles, multiply by 180
        data_len = self.length
        data_len //= self.length_divisor
        return data_len

    def load_frames_and_params(self, clip, start_frame_idx, clip_idx):
        """Load frames content"""
        fname = os.path.join(self.data_root, clip[1])
        frames_list = clip[2:]

        if not (os.path.exists(fname)):
            raise RuntimeError(f"{fname} DOES NOT EXIST***********")

        buffer = []
        camera_param_buffer = []
        features_buffer = []
        camera_cat_buffer = []
        def _read_image(path):
            try:
              img = read_image(path, ImageReadMode.RGB)
            #   img = img.unsqueeze(0)
              return img
            except RuntimeError:
              raise RuntimeError(f"IMAGE READ FAILED FOR PATH {os.path.join(fname, frames_list[fno])}***********")

        # Load camera parameter
        camera_file_name = "transforms.json"
        with open(os.path.join("/".join(fname.split(os.path.sep)[0:-2]), camera_file_name)) as json_data:
            camera_parameters_frames = json.load(json_data)["frames"]

        # print("CAMERA PARAMS:", camera_parameters_frames)

        #first_frame_camera_params = None
        prev_frame_camera_params =  None
        translation_diffs = np.array([0., 0., 0.])
        for i in range(self.new_length):
            fno = start_frame_idx + i * self.new_step

            if self.encoder_logits is not None:
                features_buffer.append(self.encoder_logits[clip_idx][fno])

            # If the frame is a .PNG
            # Read the frame using OpenCV
            img = _read_image(os.path.join(fname, frames_list[fno]))

            camera_params = np.array(camera_parameters_frames[fno]["transform_matrix"])
            # Get the parameters for the first frame to predict only relative motion
            if i == 0:
                prev_frame_camera_params = camera_params
                camera_params_cat = 0
                rotation = [0, 0, 0, 1]
                translation = [0, 0, 0]
                camera_params = torch.Tensor(rotation), torch.Tensor(translation)
            else:
                camera_params_diff = np.linalg.inv(prev_frame_camera_params) @ camera_params
                rotation = camera_params_diff[0:3, 0:3]

                quat_rotation = matrix_to_quaternion(rotation)
                translation = camera_params_diff[0:3, 3]
                translation = translation - prev_frame_camera_params[:3, 3]
                translation_diffs += np.abs(translation)
                prev_frame_camera_params = camera_params
                camera_params_cat = camera_to_category(rotation, translation)
                camera_params = (torch.Tensor(quat_rotation), torch.Tensor(translation))

            if self.is_binocular:
                fno2 = fno + self.num_frames_per_clip[clip_idx]
                img2 = _read_image(os.path.join(fname, frames_list[fno2]))
                img = torch.cat([img, img2])
            buffer.append(img)
            camera_param_buffer.append(camera_params)
            camera_cat_buffer.append(camera_params_cat)

        avg_translation_diff = torch.mean(torch.Tensor(translation_diffs)/(self.new_length-1))
        for i, param in enumerate(camera_param_buffer):
            camera_param_buffer[i] = torch.cat((param[0], param[1]/avg_translation_diff))
        if self.reverse_sequence:
            # Randomly reverse sequence
            reverse_seq = random.choice([True, False])
            if reverse_seq:
                buffer.reverse()
                features_buffer.reverse()
        return buffer, camera_param_buffer, camera_cat_buffer, features_buffer

    def load_frames(self, clip, start_frame_idx, clip_idx):
        """Load frames content"""
        fname = os.path.join(self.data_root, clip[1])
        frames_list = clip[2:]
        if not (os.path.exists(fname)):
            raise RuntimeError(f"{fname} DOES NOT EXIST***********")

        buffer = []
        features_buffer = []
        def _read_image(path):
            try:
              img = read_image(path, ImageReadMode.RGB)
              return img
            except RuntimeError:
              raise RuntimeError(f"IMAGE READ FAILED FOR PATH {os.path.join(fname, frames_list[fno])}***********")

        for i in range(self.new_length):
            fno = (start_frame_idx + i * self.new_step) % len(frames_list)
            # If the frame is a .PNG
            # Read the frame using OpenCV
            if self.encoder_logits is not None:
                features_buffer.append(self.encoder_logits[clip_idx][fno])
            img = _read_image(os.path.join(fname, frames_list[fno]))
            if self.is_binocular:
              fno2 = fno + self.num_frames_per_clip[clip_idx]
              img2 = _read_image(os.path.join(fname, frames_list[fno2]))
              img = torch.cat([img, img2])
            buffer.append(img)
        if self.reverse_sequence:
            # Randomly reverse sequence
            reversed = random.choice([True, False])
            if reversed:
                buffer.reverse()
                features_buffer.reverse()
        return buffer, features_buffer

    def make_dataset_samples(self):
        with open(self.data_list, 'r') as fopen:
            lines = fopen.readlines()
        all_seqs = [line.split() for line in lines]
        return all_seqs

class Co3dLpDataset(torch.utils.data.Dataset):
    def __init__(self,
                 root,
                 train=True,
                 datapath="",
                 transform=None,
                 lazy_init=False,
                 reverse_sequence=False,
                 train_start_frame = 0,
                 train_end_frame = 39,
                 val_start_frame = 40,
                 val_end_frame = 49):

        super(Co3dLpDataset, self).__init__()
        self.root = root
        self.train = train
        self.datapath = datapath,
        self.transform = transform
        self.lazy_init = lazy_init
        self.reverse_sequence = reverse_sequence
        self.train_start_frame = train_start_frame
        self.train_end_frame = train_end_frame
        self.val_start_frame = val_start_frame
        self.val_end_frame = val_end_frame

        if not self.lazy_init:
            self.clips = self.make_dataset_samples()
            if len(self.clips) == 0:
                raise(RuntimeError("Found 0 video clips in subfolders of: " + root + "\n"
                                   "Check your data directory (opt.data-dir)."))

        # Preprocessing the label
        all_classes = os.listdir(self.root)
        self.label_preprocessing = preprocessing.LabelEncoder()
        self.label_preprocessing.fit(np.array(all_classes))

    def __getitem__(self, index):
        sample = self.clips[index%len(self.clips)]
        image = self.load_frame(sample)

        # Adding the labels
        label = sample[0].split('/')[0]
        label = self.label_preprocessing.transform([label])

        label = torch.tensor(label)

        if self.transform:
            image = self.transform(image)

        return image, label[0]

    def __len__(self):
        return len(self.clips)

    def load_frame(self, sample):
        fname = os.path.join(self.root, sample[0])
        frames_list = sample[1:]

        if not (os.path.exists(fname)):
            print(f"Frame of {fname} does not exist")
            return []

        # Train on only training frames
        if self.train:
            frames_list = frames_list[self.train_start_frame: self.train_end_frame]

        # Validate on remaining frames
        else:
            frames_list = frames_list[self.val_start_frame: self.val_end_frame]

        selected_random_frame = random.choice(frames_list)

        img = Image.open(os.path.join(fname, selected_random_frame)).convert('RGB')  # Open the image

        return img

    def make_dataset_samples(self):
        self.datapath = self.datapath[0]
        with open(self.datapath, 'r') as fopen:
            lines = fopen.readlines()
        all_seqs = [line.split() for line in lines]
        return all_seqs

class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

class AlignmentDataset(Dataset):
    def __init__(self, numpy_file, human_results_file, label_to_index, label_dict=None, transform=None, dataset_name='co3d'):
        super(AlignmentDataset, self).__init__()
        self.image_names = []
        self.image_files = []
        self.heatmaps = []
        self.categories = []
        self.transform = transform
        self.label_to_index = label_to_index
        self.label_dict = label_dict
        self.dataset_name = dataset_name
        all_data = np.load(numpy_file, allow_pickle=True)
        human_data = np.load(human_results_file, allow_pickle=True)

        filtered_imgs = human_data['final_clickmaps'].tolist().keys()
        all_ceiling = human_data['ceiling_correlations'].tolist()
        all_null = human_data['null_correlations'].tolist()
        self.null = np.mean(all_null)
        # self.ceiling = np.mean(all_ceiling)
        self.ceiling = []
        for i, img_name in enumerate(filtered_imgs):
            cat = img_name.split('/')[0]
            if label_dict is not None:
                cat = label_dict[cat]
            if img_name in all_data.files and all_ceiling[i] > 0:
                heatmaps = all_data[img_name].tolist()['heatmap']
                self.ceiling.append(all_ceiling[i])
                self.image_files.append(all_data[img_name].tolist()['image'])
                self.heatmaps.append(heatmaps)
                self.image_names.append(img_name)
                self.categories.append(cat)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        img_label = self.image_names[idx].split('/')[0]
        img_name = self.image_names[idx]
        if self.label_dict is not None:
            img_label = self.label_dict[img_label]
        numeric_label = torch.tensor(self.label_to_index[img_label], dtype=torch.long)
        heatmap = self.heatmaps[idx]
        if self.transform:
            image = self.transform(img_file.convert("RGB"))
        else:
            image = img_file
        cat = self.categories[idx]
        ceiling = self.ceiling[idx]
        return image, heatmap, numeric_label, img_name, cat, ceiling


class NerfDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_root,
                 data_list,
                 mae_transform = True,
                 transform=None):

        super().__init__()

        self.data_root = data_root
        self.data_list = data_list
        self.transform = transform
        self.clips = self.make_dataset_samples()
        self.encoder_logits = None
        self.mae_transform = mae_transform
        labels = []
        self.num_frames_per_clip = [len(clip[2:]) for clip in self.clips]
        self.start_indices = [[0, 0]] * self.num_frames_per_clip[0]
        for i in range(len(self.num_frames_per_clip)-1):
            self.start_indices += [[i+1, self.start_indices[-1][1]+self.num_frames_per_clip[i]]] * self.num_frames_per_clip[i+1]
        for clip in self.clips:
            labels.append(clip[0])
            #labels.append(clip[0].split('/')[1])
        labels = np.unique(labels)
        self.label_preprocessing = preprocessing.LabelEncoder()
        self.label_preprocessing.fit(np.array(labels))
        if len(self.clips) == 0:
            raise(RuntimeError("Found 0 video clips in subfolders of: " + data_root + "\n"
                               "Check your data directory (opt.data-dir)."))

    def __getitem__(self, index):
        clip_idx = self.start_indices[index][0]
        start_frame_idx = index - self.start_indices[index][1]
        clip = self.clips[clip_idx]
        label = clip[0]
        label = self.label_preprocessing.transform([label])
        
        images, name = self.load_frames(clip, start_frame_idx)
        if self.mae_transform:
            process_data, _ = self.transform(([images], None)) # T,C,H,W
            process_data = process_data.squeeze()

        else:
            process_data = self.transform(images)
            process_data = process_data.squeeze()
        #process_data = process_data.view((self.new_length, 3) + process_data.size()[-2:]).transpose(0,1)  # T*C,H,W -> T,C,H,W -> C,T,H,W
        return (process_data, label, name)

    def __len__(self):
        data_len = sum(self.num_frames_per_clip)
        return data_len

    def load_frames(self, clip, start_frame_idx):
        """Load frames content"""
        fname = os.path.join(self.data_root, clip[1])
        frames_list = clip[2:]
        
        if not (os.path.exists(fname)):
            raise RuntimeError(f"{fname} DOES NOT EXIST***********")
        def _read_image(path):
            try:
                # img = read_image(path, ImageReadMode.RGB)
                img = Image.open(path).convert("RGB")
                # img = img.unsqueeze(0)
                return img
            except RuntimeError:
                raise RuntimeError(f"IMAGE READ FAILED FOR PATH {os.path.join(fname, frames_list[fno])}***********")

        img = _read_image(os.path.join(fname, frames_list[start_frame_idx]))
        return img, os.path.join(fname, frames_list[start_frame_idx])

    def make_dataset_samples(self):
        with open(self.data_list, 'r') as fopen:
            lines = fopen.readlines()
        all_seqs = [line.split() for line in lines]
        return all_seqs