import torch
import timm
import os
from src.data.datasets import build_frames_dataset
from src.util.save_features import get_features
from main_pretrain_autoreg import get_args_parser

def extract_features(args):
    device = torch.device(args.device)
    feature_path = args.features_path
    if  'autoreg_vit_small_patch16' in args.model:
        model_name = 'vit_small_patch16_224.augreg_in21k_ft_in1k'
    if 'autoreg_beit_large_patch16' in args.model:
        model_name = 'beitv2_large_patch16_224.in1k_ft_in22k_in1k'
    else:
        model_name = args.model
    model = timm.create_model(
        model_name,
        pretrained=True,
    )
    train_feature_name = args.model + '_' + args.dataset + '_logits_train.pt'
    val_feature_name = args.model + '_' + args.dataset + '_logits_val.pt'
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    mean = data_config['mean']
    std = data_config['std']
    args.mean = mean
    args.std = std
    patch_size = model.patch_embed.patch_size
    args.window_size = (args.num_frames, args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size
    model = model.to(device)
    if not (os.path.isfile(os.path.join(feature_path, train_feature_name)) and os.path.isfile(os.path.join(feature_path, val_feature_name))):
        train_dataset = build_frames_dataset(args, transforms)
        val_dataset = build_frames_dataset(args, transforms, is_train=False)
        train_features, train_labels, train_fnames = get_features(model, train_dataset, True, False, args)
        val_features, val_labels, val_fnames = get_features(model, val_dataset, False, False, args)
        train_logits, train_labels, val_logits, val_labels = train_features, train_labels, val_features, val_labels
        train_tuple = (train_logits, train_labels, train_fnames)
        val_tuple = (val_logits, val_labels, val_fnames)
        train_feature_name = args.model + '_' + args.dataset + '_logits_train.pt'
        val_feature_name = args.model + '_' + args.dataset + '_logits_val.pt'
        torch.save(train_tuple, os.path.join(args.features_path, train_feature_name))
        torch.save(val_tuple, os.path.join(args.features_path, val_feature_name))

if __name__ == '__main__':
    opts = get_args_parser().parse_args()
    torch.multiprocessing.set_sharing_strategy('file_system')
    extract_features(opts)