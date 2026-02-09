import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

import numpy as np
from torch.utils.data import ConcatDataset, Dataset, DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T


class ImageDataset(Dataset):
    def __init__(self, dataset, indices, client_idx):
        super().__init__()
        self.dataset = dataset
        self.indices = indices
        self.client_idx = client_idx

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, label = self.dataset[self.indices[idx]]
        return image, label, self.client_idx


def preprocess_dataset():
    img_size = 96
    names = ["kaggle_arima", "RIM", "REFUGE"]
    MEANS = [[0.7238, 0.3767, 0.1002], [0.5886, 0.2652, 0.1481], [0.7085, 0.4822, 0.3445]]
    STDS = [[0.1001, 0.1057, 0.0503], [0.1147, 0.0937, 0.0461], [0.1663, 0.1541, 0.1066]]

    datasets = []

    for i in range(3):
        transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=MEANS[i], std=STDS[i])
        ])
        train_path = os.path.join(base_path, "datasets", "retina_balanced", names[i], "Training")
        test_path = os.path.join(base_path, "datasets", "retina_balanced", names[i], "Testing")
        train_set = ImageFolder(train_path, transform=transform)
        test_set = ImageFolder(test_path, transform=transform)
        datasets.append(ConcatDataset([train_set, test_set]))

    return datasets


def construct_dataset(cfg, dataset, client_idx):
    num_data = len(dataset)

    indices = np.arange(num_data)
    np.random.shuffle(indices)

    num_data_train = int(cfg.task.dataset.split_ratio * num_data)

    return ImageDataset(dataset, indices[:num_data_train], client_idx), ImageDataset(dataset, indices[num_data_train:], client_idx), num_data


def load_dataset(cfg):
    assert cfg.task.training.num_clients == 3, "Retina dataset only supports 3 clients."

    datasets = preprocess_dataset()

    train_loaders, test_loaders, num_data = [], [], []

    for i in range(3):
        client_train_dataset, client_test_dataset, client_num_data = construct_dataset(cfg, datasets[i], i)

        train_loaders.append(DataLoader(client_train_dataset, batch_size=cfg.task.training.batch_size, shuffle=True))
        test_loaders.append(DataLoader(client_test_dataset, batch_size=cfg.task.training.batch_size, shuffle=False))
        num_data.append(client_num_data)

    return train_loaders, test_loaders, num_data