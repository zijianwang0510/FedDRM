import sys, os
base_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(base_path)

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset, DataLoader
from torchvision.datasets import FashionMNIST, CIFAR10, CIFAR100
from fedlab.utils.dataset.partition import CIFAR10Partitioner, CIFAR100Partitioner
import torchvision.transforms.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt


CIFAR100_FINE_TO_COARSE = [
    4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
    3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
    6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
    0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
    5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
    16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
    10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
    2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
    16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
    18, 1, 2, 15, 6, 0, 17, 8, 14, 13
]

class ImageDataset(Dataset):
    def __init__(self, cfg, dataset, idxs, client_idx):
        self.cfg = cfg
        self.dataset = dataset
        self.idxs = idxs
        self.client_idx = client_idx

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        image, label = self.dataset[self.idxs[idx]]
        transform = get_transform(self.cfg, self.client_idx)
        
        image = transform(image)
        if self.cfg.dataset == 'cifar20':
            label = CIFAR100_FINE_TO_COARSE[label]

        return image, label, self.client_idx


class ImageDataset_Global(Dataset):
    def __init__(self, cfg, dataset, idxs, client_idxs):
        self.cfg = cfg
        self.dataset = dataset
        self.idxs = idxs
        self.client_idxs = client_idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        image, label = self.dataset[self.idxs[idx]]
        client_idx = self.client_idxs[idx]
        transform = get_transform(self.cfg, client_idx)

        image = transform(image)
        if self.cfg.dataset == 'cifar20':
            label = CIFAR100_FINE_TO_COARSE[label]

        return image, label, self.client_idx


class FixedTransform(torch.nn.Module):
    def __init__(self, transform_fn, **kwargs):
        super().__init__()
        self.transform_fn = transform_fn
        self.kwargs = kwargs
    def forward(self, img):
        return self.transform_fn(img, **self.kwargs)


def get_transform_color_shift_sensitivity(cfg, idx):
    gamma_levels = [0.6, 1.0, 1.4]
    hue_levels = [-0.15, 0.0, 0.15]
    saturation_levels = [0.4, 1.0, 1.6]
    posterize_bits = 2

    gamma_idx = idx % 3
    hue_idx = (idx // 3) % 3
    saturation_idx = (idx // 9) % 3
    apply_posterize = (idx // 27) % 2

    transforms_list = []
    
    transforms_list.append(FixedTransform(F.adjust_saturation, saturation_factor=saturation_levels[saturation_idx]))
    transforms_list.append(FixedTransform(F.adjust_hue, hue_factor=hue_levels[hue_idx]))
    transforms_list.append(FixedTransform(F.adjust_gamma, gamma=gamma_levels[gamma_idx]))

    if apply_posterize == 1:
        transforms_list.append(FixedTransform(F.posterize, bits=posterize_bits))

    return transforms_list


def get_transform_color_shift_ablation(cfg, idx):
    if cfg.shift_level == 'low':
        gamma_levels = [0.9, 1.1]
        hue_levels = [-0.01, 0.01]
        saturation_levels = [0.9, 1.1]

        gamma_idx      = (idx >> 0) & 1
        hue_idx        = (idx >> 1) & 1
        saturation_idx = (idx >> 2) & 1

        gamma_factor = gamma_levels[gamma_idx]
        hue_factor = hue_levels[hue_idx]
        saturation_factor = saturation_levels[saturation_idx]

        return [FixedTransform(F.adjust_saturation, saturation_factor=saturation_factor),
                FixedTransform(F.adjust_hue, hue_factor=hue_factor),
                FixedTransform(F.adjust_gamma, gamma=gamma_factor)]
    
    elif cfg.shift_level == 'mid':
        gamma_levels = [0.75, 1.25]
        hue_levels = [-0.05, 0.05]
        saturation_levels = [0.7, 1.3]

        gamma_idx      = (idx >> 0) & 1
        hue_idx        = (idx >> 1) & 1
        saturation_idx = (idx >> 2) & 1

        gamma_factor = gamma_levels[gamma_idx]
        hue_factor = hue_levels[hue_idx]
        saturation_factor = saturation_levels[saturation_idx]

        return [FixedTransform(F.adjust_saturation, saturation_factor=saturation_factor),
                FixedTransform(F.adjust_hue, hue_factor=hue_factor),
                FixedTransform(F.adjust_gamma, gamma=gamma_factor)]
    
    elif cfg.shift_level == 'high':
        gamma_levels = [0.6, 1.4]
        hue_levels = [-0.1, 0.1]
        saturation_levels = [0.5, 1.5]

        gamma_idx      = (idx >> 0) & 1
        hue_idx        = (idx >> 1) & 1
        saturation_idx = (idx >> 2) & 1

        gamma_factor = gamma_levels[gamma_idx]
        hue_factor = hue_levels[hue_idx]
        saturation_factor = saturation_levels[saturation_idx]

        return [FixedTransform(F.adjust_saturation, saturation_factor=saturation_factor),
                FixedTransform(F.adjust_hue, hue_factor=hue_factor),
                FixedTransform(F.adjust_gamma, gamma=gamma_factor)]
    else:
        raise ValueError(f'Unsupported shift level: {cfg.shift_level}...')


def get_transform_color_shift(cfg, idx):
    if cfg.dataset == 'fmnist':
        brightness_levels = [0.5, 1.5]
        gamma_value = 2.0

        brightness_idx = (idx >> 0) & 1
        apply_invert   = (idx >> 1) & 1
        apply_gamma    = (idx >> 2) & 1

        brightness_factor = brightness_levels[brightness_idx]
        transforms_list = [FixedTransform(F.adjust_brightness, brightness_factor=brightness_factor)]

        if apply_gamma:
            transforms_list.append(FixedTransform(F.adjust_gamma, gamma=gamma_value))
        
        if apply_invert:
            transforms_list.append(F.invert)

        return transforms_list
    else:
        gamma_levels = [0.6, 1.4]
        hue_levels = [-0.1, 0.1]
        saturation_levels = [0.5, 1.5]

        gamma_idx      = (idx >> 0) & 1
        hue_idx        = (idx >> 1) & 1
        saturation_idx = (idx >> 2) & 1

        gamma_factor = gamma_levels[gamma_idx]
        hue_factor = hue_levels[hue_idx]
        saturation_factor = saturation_levels[saturation_idx]

        return [FixedTransform(F.adjust_saturation, saturation_factor=saturation_factor),
                FixedTransform(F.adjust_hue, hue_factor=hue_factor),
                FixedTransform(F.adjust_gamma, gamma=gamma_factor)]


def get_transform(cfg, idx):
    transforms_list = []

    if cfg.dataset == 'fmnist':
        transforms_list.append(T.Grayscale(num_output_channels=3))
        transforms_list.append(T.Resize([32, 32]))

    transforms_list.append(T.RandomResizedCrop((32, 32), scale=(0.9, 1.0)))
    if cfg.sensitivity_num_clients == True:
        transforms_list.extend(get_transform_color_shift_sensitivity(cfg, idx))
    elif cfg.ablation_feature_shift == True:
        transforms_list.extend(get_transform_color_shift_ablation(cfg, idx))
    else:
        transforms_list.extend(get_transform_color_shift(cfg, idx))
    transforms_list.append(T.ToTensor())
    transforms_list.append(T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    return T.Compose(transforms_list)


def label_shift(cfg, dataset):
    if cfg.dataset == 'fmnist' or cfg.dataset == 'cifar10':
        Partitioner = CIFAR10Partitioner
    elif cfg.dataset == 'cifar20' or cfg.dataset == 'cifar100':
        Partitioner = CIFAR100Partitioner

    targets = [target for _, target in dataset]

    if cfg.partition == 'dir':
        partition = Partitioner(targets=targets,
                                num_clients=cfg.num_clients,
                                balance=None,
                                partition='dirichlet', 
                                dir_alpha=cfg.dir_alpha,
                                seed=cfg.seed,
                                verbose=False)
        
        print(f"Dir with alpha = {cfg.dir_alpha}...")
    elif cfg.partition == 'shards':
        partition = Partitioner(targets=targets,
                                num_clients=cfg.num_clients,
                                balance=None,
                                partition='shards',
                                num_shards=cfg.num_clients * cfg.num_shards,
                                seed=cfg.seed,
                                verbose=False)
        
        print(f"Shards with {cfg.num_shards} shards per client...")
    else:
        raise ValueError(f'Unsupported partition mode: {cfg.partition}...')
    
    dict_users_train = {}
    dict_users_test = {}
    trainset_ratio = 0.7

    for i in range(cfg.num_clients):
        np.random.shuffle(partition.client_dict[i])
        train_size = int(len(partition.client_dict[i]) * trainset_ratio)
        dict_users_train[i] = partition.client_dict[i][:train_size]
        dict_users_test[i] = partition.client_dict[i][train_size:]

    return dict_users_train, dict_users_test


def get_idxs(cfg, dict_users):
    idxs = []
    client_idxs = []

    for i in range(cfg.num_clients):
        idxs.extend(dict_users[i])
        client_idxs.extend([i] * len(dict_users[i]))

    return idxs, client_idxs


def load_dataset(cfg, GLOBAL=False):
    if cfg.dataset == 'fmnist':
        dataset = ConcatDataset([
            FashionMNIST(root=os.path.join(base_path, 'datasets', 'FMNIST'), train=True, download=True),
            FashionMNIST(root=os.path.join(base_path, 'datasets', 'FMNIST'), train=False, download=True)
        ])
    elif cfg.dataset == 'cifar10':
        dataset = ConcatDataset([
            CIFAR10(root=os.path.join(base_path, 'datasets', 'CIFAR10'), train=True, download=True),
            CIFAR10(root=os.path.join(base_path, 'datasets', 'CIFAR10'), train=False, download=True)
        ])
    elif cfg.dataset == 'cifar20' or cfg.dataset == 'cifar100':
        dataset = ConcatDataset([
            CIFAR100(root=os.path.join(base_path, 'datasets', 'CIFAR100'), train=True, download=True),
            CIFAR100(root=os.path.join(base_path, 'datasets', 'CIFAR100'), train=False, download=True)
        ])
    else:
        raise ValueError(f'Unsupported dataset: {cfg.dataset}...')
    
    print(f"Dataset: {cfg.dataset}...")

    show_imgs(cfg, dataset, 6)

    dict_users_train, dict_users_test = label_shift(cfg, dataset)

    if GLOBAL == True:
        train_idxs, train_client_idxs = get_idxs(cfg, dict_users_train)
        test_idxs, test_client_idxs = get_idxs(cfg, dict_users_test)

        train_loader = DataLoader(ImageDataset_Global(cfg, dataset, train_idxs, train_client_idxs), batch_size=cfg.batch_size, shuffle=True)
        test_loader = DataLoader(ImageDataset_Global(cfg, dataset, test_idxs, test_client_idxs), batch_size=cfg.batch_size, shuffle=False)

        return train_loader, test_loader
    
    else:
        train_loaders = []
        test_loaders = []
        num_data = []

        for i in range(cfg.num_clients):
            train_loader = DataLoader(ImageDataset(cfg, dataset, dict_users_train[i], i), batch_size=cfg.batch_size, shuffle=True)
            test_loader = DataLoader(ImageDataset(cfg, dataset, dict_users_test[i], i), batch_size=cfg.batch_size, shuffle=False)

            num_data.append(len(dict_users_train[i]))
            train_loaders.append(train_loader)
            test_loaders.append(test_loader)

        return train_loaders, test_loaders, num_data


def show_imgs(cfg, dataset, idx):
    img, _ = dataset[idx]

    if cfg.sensitivity_num_clients == True:
        fig, axes = plt.subplots(6, 9, figsize=(20, 12))
        fig.suptitle("Covariate Shift - Sensitivity to Number of Clients", fontsize=20)

        gamma_levels = [0.6, 1.0, 1.4]
        hue_levels = [-0.15, 0.0, 0.15]
        saturation_levels = [0.4, 1.0, 1.6]

        sat_labels = ['L', 'N', 'H'] # Low, Normal, High
        hue_labels = ['R', 'N', 'G'] # Reddish, Normal, Greenish
        gam_labels = ['D', 'N', 'B'] # Dark, Normal, Bright

        for client_id in range(cfg.num_clients):
            transform = T.Compose(get_transform_color_shift_sensitivity(cfg, client_id))
            transformed_image = transform(img)

            ax = axes[client_id // 9, client_id % 9]
            ax.imshow(transformed_image)

            gamma_idx = idx % 3
            hue_idx = (idx // 3) % 3
            saturation_idx = (idx // 9) % 3
            apply_posterize = (idx // 27) % 2

            title = (f"ID:{client_id} | S:{sat_labels[saturation_idx]} "
                     f"H:{hue_labels[hue_idx]} G:{gam_labels[gamma_idx]} "
                     f"P:{'Y' if apply_posterize else 'N'}")
            ax.set_title(title, fontsize=8)
            ax.axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])

    elif cfg.ablation_feature_shift == True:
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f"Covariate Shift - Ablation ({cfg.shift_level.capitalize()} Level)", fontsize=18)

        if cfg.shift_level == 'low':
            gamma_levels = [0.9, 1.1]
            hue_levels = [-0.01, 0.01]
            saturation_levels = [0.9, 1.1]
        elif cfg.shift_level == 'mid':
            gamma_levels = [0.75, 1.25]
            hue_levels = [-0.05, 0.05]
            saturation_levels = [0.7, 1.3]
        elif cfg.shift_level == 'high':
            gamma_levels = [0.6, 1.4]
            hue_levels = [-0.1, 0.1]
            saturation_levels = [0.5, 1.5]
        else:
            raise ValueError(f'Unsupported shift level: {cfg.shift_level}...')

        for client_id in range(cfg.num_clients):
            transform = T.Compose(get_transform_color_shift_ablation(cfg, client_id))
            transformed_image = transform(img)

            ax = axes[client_id // 4, client_id % 4]
            ax.imshow(transformed_image)

            gamma_idx      = (client_id >> 0) & 1
            hue_idx        = (client_id >> 1) & 1
            saturation_idx = (client_id >> 2) & 1

            g = gamma_levels[gamma_idx]
            h = hue_levels[hue_idx]
            s = saturation_levels[saturation_idx]

            ax.set_title(f"ID:{client_id} | G:{g}, H:{h}, S:{s}", fontsize=10)
            ax.axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.94])

    elif cfg.dataset == 'fmnist':
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle("Covariate Shift - FMNIST", fontsize=18)

        brightness_levels = [0.5, 1.5]
        gamma_value = 2.0

        for client_id in range(cfg.num_clients):
            transform = T.Compose(get_transform_color_shift(cfg, client_id))
            transformed_image = transform(img)

            ax = axes[client_id // 4, client_id % 4]
            ax.imshow(transformed_image)

            brightness_idx = (client_id >> 0) & 1
            apply_invert   = (client_id >> 1) & 1
            apply_gamma    = (client_id >> 2) & 1

            b = brightness_levels[brightness_idx]
            i = "On" if apply_invert else "Off"
            g = "On" if apply_gamma else "Off"

            ax.set_title(f"ID:{client_id} | B:{b}, Inv:{i}, Gam:{g}", fontsize=10)
            ax.axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.94])

    else:
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle("Covariate Shift - CIFAR", fontsize=18)

        gamma_levels = [0.6, 1.4]
        hue_levels = [-0.1, 0.1]
        saturation_levels = [0.5, 1.5]

        for client_id in range(cfg.num_clients):
            transform = T.Compose(get_transform_color_shift(cfg, client_id))
            transformed_image = transform(img)

            ax = axes[client_id // 4, client_id % 4]
            ax.imshow(transformed_image)

            gamma_idx      = (client_id >> 0) & 1
            hue_idx        = (client_id >> 1) & 1
            saturation_idx = (client_id >> 2) & 1

            g = gamma_levels[gamma_idx]
            h = hue_levels[hue_idx]
            s = saturation_levels[saturation_idx]

            ax.set_title(f"ID:{client_id} | G:{g}, H:{h}, S:{s}", fontsize=10)
            ax.axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.94])

    plt.show()