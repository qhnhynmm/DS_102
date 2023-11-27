import torch
import torchvision
import torchvision.transforms as transforms
from typing import List, Dict, Optional
from torch.utils.data import DataLoader, Dataset,random_split
from PIL import Image
class IMG_Dataset(Dataset):
    def __init__(self, path, image_W, image_H, aug=True):
        self.image_H = image_H
        self.image_W = image_W
        self.aug = aug
        if self.aug:
            self.transform = transforms.Compose([
                transforms.Resize((self.image_H, self.image_W)),
                transforms.RandomCrop(size=(self.image_H, self.image_W), padding=4),       
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((self.image_H, self.image_W)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        self.data=self.load_data(path)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        return {'image':image,'label':label}

    def load_data(self, data_path):
        dataset = torchvision.datasets.ImageFolder(
            root=data_path,
            transform=self.transform
        )
        return dataset


class Load_data:
    def __init__(self, config):
        self.train_path=config['train_path']
        self.test_path=config['test_path']

        self.train_batch=config['train_batch']
        self.dev_batch=config['dev_batch']
        self.test_batch=config['test_batch']

        self.image_H = config['image_H']
        self.image_W = config['image_W']
        self.aug = config['aug']
        self.num_worker=config['num_worker']

    def load_train_dev(self):
        train_dev_set=IMG_Dataset(self.train_path,self.image_W,self.image_H,self.aug)
        dataset_size = len(train_dev_set)
        train_size = int(0.9 * dataset_size)
        dev_size = dataset_size - train_size
        train_set, dev_set = random_split(train_dev_set, [train_size, dev_size])
        train_loader = DataLoader(train_set, batch_size=self.train_batch, num_workers=self.num_worker,shuffle=True)
        dev_loader = DataLoader(dev_set, batch_size=self.dev_batch, num_workers=self.num_worker,shuffle=True)
        return train_loader, dev_loader
    
    def load_test(self):
        test_set=IMG_Dataset(self.test_path,self.image_W,self.image_H,self.aug)
        test_loader = DataLoader(test_set, batch_size=self.dev_batch, num_workers=self.num_worker,shuffle=False)
        return test_loader
        

