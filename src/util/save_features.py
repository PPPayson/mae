import torch
import utils
import os
from tqdm import tqdm
from models_2D import LinearModel
from data.co3d_dataset import EmbeddingDataset
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
from data.datasets import build_frames_dataset
from typing import Iterable


def extract_features(model: torch.nn.Module, data_loader: Iterable, device: torch.device, pool: bool):
    model.eval()
    model.to(device)
    features = []
    labels_list = []
    for i, data in enumerate(tqdm(data_loader)):
        with torch.no_grad():
            images, labels = data
            images = images.to(device)
            preds, _, _ = model.forward_encoder(images, 0)
            if len(preds.shape) > 2:
                preds = torch.mean(preds, dim=1)
            features.append(preds.cpu())
            labels_list.append(labels.cpu())

    features = torch.cat(features)
    labels = torch.cat(labels_list).squeeze()
    return features, labels

def save_features(encoder, dataset, args):
    encoder.eval()
    #reverse_sequence = dataset.reverse_sequence
    dataset.reverse_sequence = False
    data_loader = torch.utils.data.DataLoader(
        dataset, shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        persistent_workers=True,
    )
    encoder_activations = []
    labels = []
    fnames = []
    with torch.cuda.amp.autocast(enabled=True):
        for step, batch in enumerate(tqdm(data_loader)):
            videos, label, fname = batch[0].to(args.device), batch[1], batch[2]
            # B, C, T, H, W = videos.shape
            # videos = videos.movedim(1, 2).reshape(B*T, C, H, W)
            labels.append(label)
            fnames += fname
            #activation = torch.mean(encoder(videos, f2d=True).detach().cpu(), dim=1)
            #activation = encoder(videos, f2d=True).detach().cpu()
            activation = encoder(videos).detach().cpu()
            if len(activation.shape) > 2:
                activation = torch.mean(activation, dim=1)
            encoder_activations.append(activation)
    encoder_activations = torch.cat(encoder_activations)
    #dataset.reverse_sequence = reverse_sequence
    return encoder_activations, torch.cat(labels), fnames

def get_features(encoder, dataset, train, args):
    feature_path = args.features_path
    if train:
        feature_name = args.model + '_train.pt'
    else:
        feature_name = args.model + '_val.pt'
    if not os.path.isfile(os.path.join(feature_path, feature_name)):
        encoder_activations = save_features(encoder, dataset, args)
        torch.save(encoder_activations, os.path.join(feature_path, feature_name))
    else:
        encoder_activations = torch.load(os.path.join(feature_path, feature_name))
    return encoder_activations


def get_logits(train_features, train_labels, train_fnames, val_features, val_labels, val_fnames, args):
    train_features = torch.squeeze(train_features)
    val_features = torch.squeeze(val_features)
    train_labels = train_labels.numpy().squeeze().flatten().tolist()
    val_labels = val_labels.numpy().squeeze().flatten().tolist()
    train_dataset = EmbeddingDataset(train_features, train_labels)
    val_dataset = EmbeddingDataset(val_features, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)


    linear_model = LinearModel(input_dim=train_features.shape[-1], num_classes = len(set(train_labels.numpy()))).to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(linear_model.parameters(), lr=0.0001)

    best_acc = 0
    for e in tqdm(range(30)):
        linear_model.train()
        train_loss = 0
        for batch in train_loader:
            embeddings, labels = batch
            embeddings = embeddings.to(args.device)
            labels = labels.to(args.device).type(torch.long)
            preds = linear_model(embeddings)
            loss = criterion(preds, labels)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        avg_train_loss = train_loss / len(train_loader)
        print("train_loss", avg_train_loss)
        linear_model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                embeddings, labels = batch
                embeddings = embeddings.to(args.device)
                labels = labels.to(args.device)
                preds = linear_model(embeddings)
                loss = criterion(preds, labels)
                val_loss += loss.item()
                _, predicted = torch.max(preds.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        avg_val_loss = val_loss / len(val_loader)
        acc = 100 * correct/total
        best_acc = max(acc, best_acc)
        print("val_loss", avg_val_loss)
        print("Acc", acc)
        print("Best_acc", best_acc)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    train_logits = []
    val_logits = []
    train_labels = []
    val_labels = []
    linear_model.eval()
    with torch.no_grad():
        for batch in train_loader:
            embeddings, labels = batch
            embeddings = embeddings.to(args.device)
            labels = labels.to(args.device).type(torch.long)
            preds = linear_model(embeddings)
            train_logits.append(preds)
            train_labels.append(labels)
        for batch in val_loader:
            embeddings, labels = batch
            embeddings = embeddings.to(args.device)
            labels = labels.to(args.device)
            preds = linear_model(embeddings)
            val_logits.append(preds)
            val_labels.append(labels)
    train_logits = torch.cat(train_logits)
    train_labels = torch.cat(train_labels)
    val_logits = torch.cat(val_logits)
    val_labels = torch.cat(val_labels)
    train_tuple = (train_logits, train_labels, train_fnames)
    val_tuple = (val_logits, val_labels, val_fnames)
    train_feature_name = args.model + '_' + args.dataset + '_logits_train.pt'
    val_feature_name = args.model + '_' + args.dataset + '_logits_val.pt'
    torch.save(train_tuple, os.path.join(args.features_path, train_feature_name))
    torch.save(val_tuple, os.path.join(args.features_path, val_feature_name))
    return train_logits, train_labels, val_logits, val_labels

def load_logits(encoder, args):
    feature_path = args.features_path
    if hasattr(encoder, "model_name"):
        model_name = encoder.model_name
    else:
        model_name = args.encoder_name
    train_feature_name = model_name + '_' + args.dataset + '_logits_train.pt'
    val_feature_name = model_name + '_' + args.dataset + '_logits_val.pt'
    if not (os.path.isfile(os.path.join(feature_path, train_feature_name)) and os.path.isfile(os.path.join(feature_path, val_feature_name))):
        train_dataset = build_frames_dataset(args)
        val_dataset = build_frames_dataset(args, is_train=False)
        train_features, train_labels, train_fnames = get_features(encoder, train_dataset, True, args)
        val_features, val_labels, val_fnames = get_features(encoder, val_dataset, False, args)
        train_logits, train_labels, val_logits, val_labels = get_logits(train_features, train_labels, train_fnames, val_features, val_labels, val_fnames, args)
    else:
        train_logits, train_labels, train_fnames = torch.load(os.path.join(feature_path, train_feature_name))
        val_logits, val_labels, val_fnames = torch.load(os.path.join(feature_path, val_feature_name))
    return train_logits, train_labels, train_fnames, val_logits, val_labels, val_fnames
    