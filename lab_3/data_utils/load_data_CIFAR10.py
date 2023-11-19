import os
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets

class CIFAR10Dataset(Dataset):
    def __init__(self, root, train=True, transform=None, download=True):
        if download:
            self.download(root)
        self.data = datasets.CIFAR10(root=root, train=train, transform=transform)

    def download(self, root):
        datasets.CIFAR10(root=root, train=True, download=True)
        datasets.CIFAR10(root=root, train=False, download=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]

        # Check if the image is already a tensor
        if not isinstance(img, torch.Tensor):
            img = transforms.ToTensor()(img)

        return {'image': img, 'label': torch.tensor(label)}

class CIFAR10LoadData:
    def __init__(self, config):
        self.root = config['root']
        self.train_batch = config['train_batch']
        self.val_batch = config['val_batch']
        self.test_batch = config['test_batch']

    def load_train_dev(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        train_dev_set = CIFAR10Dataset(root=self.root, train=True, transform=transform, download=True)
        dataset_size = len(train_dev_set)
        train_size = int(0.9 * dataset_size)
        dev_size = dataset_size - train_size
        train_set, val_set = random_split(train_dev_set, [train_size, dev_size])

        train_loader = DataLoader(train_set, batch_size=self.train_batch, num_workers=2, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=self.val_batch, num_workers=2, shuffle=True)
        return train_loader, val_loader

    def load_test(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        test_set = CIFAR10Dataset(root=self.root, train=False, transform=transform, download=True)
        test_loader = DataLoader(test_set, batch_size=self.test_batch, num_workers=2, shuffle=False)
        return test_loader
