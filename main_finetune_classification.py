import os 
import argparse
import timm
import torch
import src.util.misc as misc
from src.models.backbones.models_encoder import LinearModel, FullModel
import numpy as np
import random
import wandb
from src.util.optim_factory import create_optimizer
from src.data.datasets import build_frames_dataset, DataAugmentationForVideoMAE, build_pretraining_dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from scipy.stats import spearmanr
from PIL import Image
from src.engine_eval import eval_alignment
import types
import torchvision.datasets as tv_datasets
from torchvision import transforms
from src.data.co3d_dataset import Co3dLpDataset, AlignmentDataset, NerfDataset, EmbeddingDataset
from tqdm import tqdm
from src.util.metrics import create_label_index_map
from einops import rearrange
torch.multiprocessing.set_sharing_strategy('file_system')
def get_arg_parser():
    parser = argparse.ArgumentParser("Finetune with Co3D data", allow_abbrev=False)
    parser.add_argument('--ckpt_path', required=False, type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--model', default='vit_small_patch16_224.augreg_in21k_ft_in1k')
    parser.add_argument('--drop_rate', default=0., type=float)
    parser.add_argument('--input_size', default=224, type=int,
                        help='videos input size for backbone')
    parser.add_argument('--mask_type', default='tube', choices=['random', 'tube', 'causal', 'causal_interpol'],
                        type=str, help='masked strategy of video tokens/patches')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='ratio of the visual tokens/patches need be masked')
    parser.add_argument('--num_frames', type=int, default= 4)
    parser.add_argument('--sampling_rate', type=int, default= 4)
    parser.add_argument('--data_length_divisor', default=1, type=int)

    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--dataset', default='co3d', type=str, choices=['co3d', 'mvimgnet', 'co3d_mvimgnet', 'imgnet', 'co3d_gs'])
    parser.add_argument('--data_root', required=True, type=str)
    parser.add_argument('--data_path', default='/path/to/list_kinetics-400', type=str,
                        help='dataset path')
    parser.add_argument('--data_val_path', default='/path/to/list_kinetics-400', type=str,
                        help='eval dataset path')
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument('--num_classes', default=51, type=int)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--custom_head', default=False, action='store_true')
    parser.add_argument("--not_pretrained", default=False, action='store_true')
    parser.add_argument('--wandb', default=False, action="store_true")
    parser.add_argument('--seed', default=84)
    parser.add_argument('--stats_json', default='./assets/click_stats.json', type=str)
    parser.add_argument('--clickmaps_path', default='./assets/co3d_val_processed.npz', type=str)
    parser.add_argument('--alignments_json', default='./assets/alignments.json', type=str)
    parser.add_argument('--clickmaps_human_path', default='./assets/human_ceiling_split_half_co3d_val.npz', type=str)
    parser.add_argument('--imgnet_clickmaps_path', default='./assets/jay_imagenet_for_co3d_val_0.1_processed.npz', type=str)
    parser.add_argument('--imgnet_clickmaps_human_path', default='./assets/human_ceiling_split_half_jay_imagenet_for_co3d_val_0.1.npz', type=str)
    parser.add_argument('--imgnet2co3d_label', default='./assets/synset_to_co3d.npy', type=str)
    parser.add_argument('--max_subjects', default=-1, type=int)
    # Optimizer parameters
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    parser.add_argument('--categorical_camera', default=False, action='store_true', help='Use 64 categories for camera poses')

    # Camera Parameters
    parser.add_argument('--camera_param_dim', type=int, default=7)
    parser.add_argument('--camera_params', action='store_true', default=False, help='Train with Camera Parameters in the Decoder')
    return parser

def create_label_index_map_imgnet(data_path):
    label_to_index_map = {}
    categories = sorted(os.listdir(os.path.join(data_path, 'train')))
    for i, c in enumerate(categories):
        label_to_index_map[c] = i
    return label_to_index_map

def build_eval_loader(args, transform=None, return_all=False):
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
        print("Using default transform for alignment eval")
        transform = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=args.mean, std=args.std)])
    else:
        print(transform)

    if args.dataset != 'imgnet':
        label_dict = None
    else:
        label_dict = np.load(args.imgnet2co3d_label, allow_pickle=True).item()

    co3d_dataset_test = AlignmentDataset(numpy_file=test_data_path, human_results_file=test_human_results, label_to_index=label_to_index_map, label_dict=label_dict,
                         transform=transform)
    co3d_dataloader_test = DataLoader(co3d_dataset_test, batch_size=1, num_workers=args.num_workers, shuffle=False)
    if args.dataset != 'imgnet':
        label_dict = np.load(args.imgnet2co3d_label, allow_pickle=True).item()
    else:
        label_dict = None
        label_to_index_map = create_label_index_map_imgnet(args.data_root)
    imgnet_dataset_test = AlignmentDataset(numpy_file=test_imgnet_path, human_results_file=test_imgnet_human_resutls,
                                            label_to_index=label_to_index_map, label_dict=label_dict, transform=transform, 
                                            dataset_name='imgnet') #max_subjects=args.max_subjects)
    imgnet_dataloader_test = DataLoader(imgnet_dataset_test, batch_size=1, num_workers=args.num_workers, shuffle=False)

    return co3d_dataloader_test, imgnet_dataloader_test


def get_dataloaders(model, args):
    data_config = timm.data.resolve_model_data_config(model)
    #transform = utils.get_transform_center_crop(data_config)
    transform_train = timm.data.create_transform(**data_config, is_training=True)
    transform_val = timm.data.create_transform(**data_config, is_training=False)
    if args.dataset == 'imgnet':
        train_dataset = tv_datasets.ImageFolder(os.path.join(args.data_root, 'train'), transform=transform_train)
        val_dataset = tv_datasets.ImageFolder(os.path.join(args.data_root, 'val'), transform=transform_val)
    elif args.dataset == 'co3d_gs':
        args.window_size=(args.num_frames, 14, 14)
        args.patch_size=(16, 16)
        train_dataset = build_pretraining_dataset(args)
        val_dataset = build_frames_dataset(args, is_train=False, transform=transform_val)
        # val_dataset = build_pretraining_dataset(args, is_train=False)        
    else:
        train_dataset = build_frames_dataset(args, transform=transform_train)
        val_dataset = build_frames_dataset(args, is_train=False, transform=transform_val)

    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True, seed=args.seed
        )
        print("Sampler_train = %s" % str(sampler_train))
        sampler_val = torch.utils.data.DistributedSampler(
            val_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True, seed=args.seed
        )
    else:
        sampler_train = torch.utils.data.RandomSampler(train_dataset)
        sampler_val = torch.utils.data.RandomSampler(val_dataset)
        num_tasks=1

    train_dataloader = DataLoader(train_dataset, sampler=sampler_train, batch_size=args.batch_size,
                                num_workers=args.num_workers, pin_memory=True, drop_last=True,
                                persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, sampler=sampler_val, batch_size=args.batch_size,
                                num_workers=args.num_workers, pin_memory=True, drop_last=False,
                                persistent_workers=True)
    data_config = timm.data.resolve_model_data_config(model)
    test_dataloader, test_imgnet_loader = build_eval_loader(args, None, False)
    num_training_steps_per_epoch = len(train_dataset) // args.batch_size // num_tasks

    return train_dataloader, val_dataloader, test_dataloader, test_imgnet_loader, num_training_steps_per_epoch

def extract_features(model, data_loader, device):
    model.eval()
    model.to(device)
    features = []
    labels_list = []
    for i, data in enumerate(tqdm(data_loader)):
        with torch.no_grad():
            images, labels, _ = data
            images = images.to(device)
            #preds = model(images, f2d=True, pool=pool)
            preds = model.forward_head(model.forward_features(images), pre_logits=True)
            if len(preds.shape) > 2:
                preds = torch.mean(preds, dim=1)
            features.append(preds.cpu())
            labels_list.append(labels.cpu())

    features = torch.cat(features)
    labels = torch.cat(labels_list).squeeze()
    return features, labels
    
def pretrain_head(model, train_loader, val_loader, epochs, device, args):
    train_features, train_labels = extract_features(model, train_loader, device)
    val_features, val_labels = extract_features(model, val_loader, device)
    metric_logger = misc.MetricLogger(delimiter='  ')
    header = f'Pretrain Head'
    print_freq = 1

    train_dataset = EmbeddingDataset(train_features, train_labels)
    val_dataset = EmbeddingDataset(val_features, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.lr, weight_decay=1e-4)

    for e in metric_logger.log_every(range(epochs), print_freq, header):
        metric_logger.update(epoch=e)
        model.head.train()
        train_loss = 0
        for batch in train_loader:
            embeddings, labels = batch
            embeddings = embeddings.to(device)
            labels = labels.to(device).type(torch.long)
            preds = model.head(embeddings)
            loss = criterion(preds, labels)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        avg_train_loss = train_loss/len(train_loader)

        model.head.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                embeddings, labels = batch
                embeddings = embeddings.to(device)
                labels = labels.to(device)
                preds = model.head(embeddings)
                loss = criterion(preds, labels)
                val_loss += loss.item()
                _, predicted = torch.max(preds.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        avg_val_loss = val_loss / len(val_loader)
        acc = 100 * correct/total
        metric_logger.update(train_loss=avg_train_loss)
        metric_logger.update(val_loss=avg_val_loss)
        metric_logger.update(acc=acc)
        metric_logger.synchronize_between_processes()

def train(model, model_without_ddp, train_loader, val_loader, test_loader, test_imgnet_loader, optimizer, log_writer, device, epochs, start_epoch, args):
    best_acc_train = 0
    best_acc_val = 0
    best_acc_test = 0
    best_alignment = 0
    best_imgnet_align = 0
    best_imgnet_acc = 0
    best_epoch = 0
    criterion = torch.nn.CrossEntropyLoss()
    if log_writer is not None:
        log_writer.set_step()
    
    acc_test, alignment_scores, null_scores = eval_alignment(model_without_ddp, test_loader, device, log_writer, list(range(10)), args)
    imgnet_acc, imgnet_align, imgnet_null = eval_alignment(model_without_ddp, test_imgnet_loader, device, log_writer, list(range(10)), args)

    alignment = np.mean(alignment_scores['full_spearman'])
    imgnet_alignment = np.mean(imgnet_align['full_spearman'])
    with torch.no_grad():
      acc_val, loss_val = evaluate(model, val_loader, criterion, -1, device, log_writer)
    print(f"Accuracy: {acc_val} Co3D Alignment: {np.mean(alignment_scores['full_spearman'])} ImgNet Alignment: {np.mean(imgnet_align['full_spearman'])}")
    print(f"Co3D Unnorm: {np.mean(alignment_scores['unnorm_spearman'])} ImgNet Unnorm: {np.mean(imgnet_align['unnorm_spearman'])}")
    print(f"Co3D Topk: {np.mean(alignment_scores['topk_spearman'])} ImgNet TopK: {np.mean(imgnet_align['topk_spearman'])}")
    if log_writer is not None:
        log_writer.update(test_acc=acc_test, head='co3d_eval')
        log_writer.update(full_spearman=np.mean(alignment_scores['full_spearman']), head='co3d_eval')
        log_writer.update(full_unnorm=np.mean(alignment_scores['unnorm_spearman']), head='co3d_eval')
        log_writer.update(full_null=np.mean(null_scores['full_spearman']), head='co3d_eval')
        log_writer.update(full_null_unnorm=np.mean(null_scores['unnorm_spearman']), head='co3d_eval')

        log_writer.update(topk_spearman=np.mean(alignment_scores['topk_spearman']), head='co3d_eval')
        log_writer.update(topk_unnorm=np.mean(alignment_scores['unnorm_topk']), head='co3d_eval')
        log_writer.update(topk_null=np.mean(null_scores['topk_spearman']), head='co3d_eval')
        log_writer.update(topk_null_unnorm=np.mean(null_scores['unnorm_topk']), head='co3d_eval')


        log_writer.update(test_acc=imgnet_acc, head='imgnet_eval')
        log_writer.update(full_spearman=np.mean(imgnet_align['full_spearman']), head='imgnet_eval')
        log_writer.update(full_unnorm=np.mean(imgnet_align['unnorm_spearman']), head='imgnet_eval')
        log_writer.update(full_null=np.mean(imgnet_null['full_spearman']), head='imgnet_eval')
        log_writer.update(full_null_unnorm=np.mean(imgnet_null['unnorm_spearman']), head='imgnet_eval')

        log_writer.update(topk_spearman=np.mean(imgnet_align['topk_spearman']), head='imgnet_eval')
        log_writer.update(topk_unnorm=np.mean(imgnet_align['unnorm_topk']), head='imgnet_eval')
        log_writer.update(topk_null=np.mean(imgnet_null['topk_spearman']), head='imgnet_eval')
        log_writer.update(topk_null_unnorm=np.mean(imgnet_null['unnorm_topk']), head='imgnet_eval')
            

    best_acc_val = acc_val
    best_acc_test = acc_test
    best_alignment = alignment
    best_imgnet_align = imgnet_alignment
    best_imgnet_acc = imgnet_acc
    params = model.parameters()
    for epoch in range(epochs):
        model.train()
        header = f'Epoch: [{epoch}]'
        metric_logger = misc.MetricLogger(delimiter="  ")
        print_freq = 50        
        for i, batch in enumerate(metric_logger.log_every(train_loader, print_freq, header)):
            if args.dataset == 'co3d_gs':
                imgs, labels = batch[0].to(args.device), batch[-1].to(args.device)
                labels = torch.squeeze(labels)
                imgs = rearrange(imgs, 'b c t h w -> (b t) c h w')
                labels = rearrange(labels, 'b t -> (b t)')
            else:
                imgs, labels = batch[0].to(args.device), batch[1].to(args.device)
                labels = torch.squeeze(labels)
            optimizer.zero_grad()

            preds = model(imgs)
            loss = criterion(preds, labels)
            loss.backward()
            if args.clip_grad is not None and args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(params, args.clip_grad)
            optimizer.step()
            
            acc = misc.accuracy(preds, labels)[0].item()
            torch.cuda.synchronize()
            metric_logger.update(acc=acc)
            metric_logger.update(loss=loss)
        metric_logger.synchronize_between_processes()
        if log_writer is not None:
            log_writer.set_step((start_epoch + epoch+1)*len(train_loader))
            log_writer.update(loss=metric_logger.loss.global_avg, head="train")
            log_writer.update(accuracy=metric_logger.acc.global_avg, head="train", commit=True)
        avg_acc = metric_logger.acc.global_avg
        avg_loss = metric_logger.loss.global_avg
        with torch.no_grad():
            if log_writer is not None:
                log_writer.set_step()
            acc_val, loss_val = evaluate(model, val_loader, criterion, epoch, device, log_writer)
        acc_test, alignment_scores, null_scores = eval_alignment(model_without_ddp, test_loader, device, log_writer, list(range(10)), args)
        imgnet_acc, imgnet_align, imgnet_null = eval_alignment(model_without_ddp, test_imgnet_loader, device, log_writer, list(range(10)), args)
        
        alignment = np.mean(alignment_scores['full_spearman'])
        imgnet_alignment = np.mean(imgnet_align['full_spearman'])
        print(f"Accuracy: {acc_val} Co3D Alignment: {np.mean(alignment_scores['full_spearman'])} ImgNet Alignment: {np.mean(imgnet_align['full_spearman'])}")
        print(f"Co3D Unnorm: {np.mean(alignment_scores['unnorm_spearman'])} ImgNet Unnorm: {np.mean(imgnet_align['unnorm_spearman'])}")
        print(f"Co3D Topk: {np.mean(alignment_scores['topk_spearman'])} ImgNet TopK: {np.mean(imgnet_align['topk_spearman'])}")
        if log_writer is not None:
            log_writer.update(test_acc=acc_test, head='co3d_eval')
            log_writer.update(full_spearman=np.mean(alignment_scores['full_spearman']), head='co3d_eval')
            log_writer.update(full_unnorm=np.mean(alignment_scores['unnorm_spearman']), head='co3d_eval')
            log_writer.update(full_null=np.mean(null_scores['full_spearman']), head='co3d_eval')
            log_writer.update(full_null_unnorm=np.mean(null_scores['unnorm_spearman']), head='co3d_eval')

            log_writer.update(topk_spearman=np.mean(alignment_scores['topk_spearman']), head='co3d_eval')
            log_writer.update(topk_unnorm=np.mean(alignment_scores['unnorm_topk']), head='co3d_eval')
            log_writer.update(topk_null=np.mean(null_scores['topk_spearman']), head='co3d_eval')
            log_writer.update(topk_null_unnorm=np.mean(null_scores['unnorm_topk']), head='co3d_eval')


            log_writer.update(test_acc=imgnet_acc, head='imgnet_eval')
            log_writer.update(full_spearman=np.mean(imgnet_align['full_spearman']), head='imgnet_eval')
            log_writer.update(full_unnorm=np.mean(imgnet_align['unnorm_spearman']), head='imgnet_eval')
            log_writer.update(full_null=np.mean(imgnet_null['full_spearman']), head='imgnet_eval')
            log_writer.update(full_null_unnorm=np.mean(imgnet_null['unnorm_spearman']), head='imgnet_eval')

            log_writer.update(topk_spearman=np.mean(imgnet_align['topk_spearman']), head='imgnet_eval')
            log_writer.update(topk_unnorm=np.mean(imgnet_align['unnorm_topk']), head='imgnet_eval')
            log_writer.update(topk_null=np.mean(imgnet_null['topk_spearman']), head='imgnet_eval')
            log_writer.update(topk_null_unnorm=np.mean(imgnet_null['unnorm_topk']), head='imgnet_eval')
            if acc_val > best_acc_val:
                best_epoch = epoch
                best_acc_val = acc_val
                best_acc_test = acc_test
                best_alignment = alignment
                best_imgnet_align = imgnet_alignment
                best_imgnet_acc = imgnet_acc
                best_acc_train = avg_acc
                torch.save(model.state_dict(), f'{args.output_dir}/best_val.ckpt')
    return {'acc_train': best_acc_train, 'acc_val': best_acc_val, 'acc_test': best_acc_test,
            'epoch': best_epoch, 'align': best_alignment, 'imgnet_align': best_imgnet_align, 
            'imgnet_acc': best_imgnet_acc}

def evaluate(model, val_loader, criterion, epoch, device, log_writer):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = f'Val Epoch: [{epoch}]'
    print_freq = 50
    accs = []
    num_correct = 0
    total_num = 0
    for i, batch in enumerate(metric_logger.log_every(val_loader, print_freq, header)):
        imgs, labels = batch[0].to(device), batch[1].to(device)
        labels = torch.squeeze(labels)

        preds = model(imgs)
        loss = criterion(preds, labels)
        acc = misc.accuracy(preds, labels)[0].item()
        accs.append(acc)
        num_correct += torch.sum(torch.argmax(preds, dim=1) == labels)
        total_num += len(imgs)
        metric_logger.update(acc=acc)
        metric_logger.update(loss=loss)
    metric_logger.synchronize_between_processes()
    
    avg_acc = metric_logger.acc.global_avg
    avg_loss = metric_logger.loss.global_avg
    print(np.mean(accs), avg_acc, num_correct/float(total_num))
    if log_writer is not None:
        log_writer.update(accuracy=avg_acc, head='val')
        log_writer.update(epoch=epoch, head='val')
        log_writer.update(loss=avg_loss, head='val', commit=False)

    return avg_acc, avg_loss

def main(args):
    misc.init_distributed_mode(args)
    device = torch.device(args.device)

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    os.makedirs(os.path.join(args.output_dir, args.model), exist_ok=True)
    args.output_dir = os.path.join(args.output_dir, args.model)
    if args.custom_head:
        model = timm.create_model(args.model, pretrained=(not args.not_pretrained), num_classes=0, drop_rate=args.drop_rate, drop_path_rate=args.drop_path_rate)
        linear_model = LinearModel(model.embed_dim, args.num_classes) 
        model = FullModel(model, linear_model)
    else:
        model = timm.create_model(args.model, pretrained=(not args.not_pretrained), num_classes=args.num_classes, drop_rate=args.drop_rate)
        # model = timm.create_model(args.model, pretrained=(not args.not_pretrained), drop_rate=args.drop_rate)

    # model.forward_grad = forward_grad
    # model.forward_grad = types.MethodType(forward_grad, model)
    global_rank = misc.get_rank()
    if global_rank == 0 and args.wandb:
        wandb.login(key="50923956f844f2494110e5b6c020d31fa1072149")
        log_writer = misc.WandBLogger(log_dir=os.path.join(args.output_dir, args.model), args=args, project='multiviewMAE')
    else:
        log_writer = None
    config = timm.data.resolve_model_data_config(model)
    args.mean = config['mean']
    args.std = config['std']
    #patch_size = model.patch_embed.patch_size
    #print(f"Patch size = {patch_size}")
    #args.window_size = (1, args.input_size // patch_size[0], args.input_size // patch_size[1])
    #args.patch_size = patch_size
    # args.std = config['std']
    total_batch_size = args.batch_size * misc.get_world_size()
    args.lr = args.lr * total_batch_size / 256

    train_loader, val_loader, test_loader, test_imgnet_loader, num_training_steps_per_epoch = get_dataloaders(model, args)
    args.warmup_steps = args.warmup_epochs * num_training_steps_per_epoch
    #args.schedule_free = True
    #args.opt = 'adamWScheduleFree'
    args.schedule_free = False
    args.opt = 'adam'
    optimizer = create_optimizer(args, model, filter_bias_and_bn=False)
    model_without_ddp = model
    model = model.to(device)
    # pretrain_head(model, train_loader, val_loader, 50, device, args)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        if args.ckpt_path != None:
            model.load_state_dict(torch.load(args.ckpt_path))
    else:
        if args.ckpt_path != None:
            ckpt_dict = torch.load(args.ckpt_path)
            for k in list(ckpt_dict.keys()):
                if 'module.' in k:
                    n_k = k.replace('module.', '')
                    ckpt_dict[n_k] = ckpt_dict[k]
                    del ckpt_dict[k]
            model.load_state_dict(ckpt_dict)
    results = train(model, model_without_ddp, train_loader, val_loader, test_loader, test_imgnet_loader, optimizer, log_writer, device, args.epochs, 0, args)
    for key in results:
        print(f'{key}: {results[key]}')

if __name__ == "__main__":
    args = get_arg_parser().parse_args()
    main(args)