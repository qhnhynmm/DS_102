# import torch
# import torchvision
# import torchvision.transforms as transforms
# from typing import List, Dict, Optional
# from torch.utils.data import DataLoader, Dataset
# from PIL import Image
# class IMG_Dataset(Dataset):
#     def __init__(self, path, image_W, image_H, aug=True):
#         self.image_H = image_H
#         self.image_W = image_W
#         self.aug = aug
#         if self.aug:
#             self.transform = transforms.Compose([
#                 transforms.Resize((self.image_H, self.image_W)),
#                 transforms.RandomCrop(size=(self.image_H, self.image_W), padding=4),       
#                 transforms.RandomRotation(10),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#             ])
#         else:
#             self.transform = transforms.Compose([
#                 transforms.Resize((self.image_H, self.image_W)),
#                 transforms.ToTensor(),
#             ])
#         self.data=self.load_data(path)
#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         img, label = self.data[idx]
#         return img, label

#     def load_data(self, data_path):
#         dataset = torchvision.datasets.ImageFolder(
#             root=data_path,
#             transform=self.transform
#         )
#         return dataset


# class Load_data:
#     def __init__(self, config):
#         self.train_path=config['train_path
#         self.dev_path=config['dev_path
#         self.test_path=config['test_path

#         self.train_batch=config['train_batch
#         self.dev_batch=config['dev_batch
#         self.test_batch=config['test_batch

#         self.image_H = config['image_H
#         self.image_W = config['image_W
#         self.aug = config['aug
#         self.num_worker=config['num_worker

#     def load_train_dev(self):
#         train_set=IMG_Dataset(self.train_path,self.image_W,self.image_H,self.aug)
#         dev_set=IMG_Dataset(self.dev_path,self.image_W,self.image_H,self.aug)
#         train_loader = DataLoader(train_set, batch_size=self.train_batch, num_workers=self.num_worker,shuffle=True)
#         dev_loader = DataLoader(dev_set, batch_size=self.dev_batch, num_workers=self.num_worker,shuffle=True)
#         return train_loader, dev_loader
    
#     def load_test(self):
#         test_set=IMG_Dataset(self.test_path,self.image_W,self.image_H,self.aug)
#         test_loader = DataLoader(test_set, batch_size=self.dev_batch, num_workers=self.num_worker,shuffle=False)
#         return test_loader
        

import struct
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import idx2numpy

class MNISTDataset(Dataset):
    def __init__(self, images_file, labels_file):
        self.data=self.load_data(images_file,labels_file)
    
    def load_data(self, images_file, labels_file):
        images = idx2numpy.convert_from_file(images_file)
        labels = idx2numpy.convert_from_file(labels_file)

        images = torch.tensor(images, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
        image_labels=[]
        for image,label in zip(images,labels):
            image_labels.append({'image':image,'label':label})
        return image_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class Load_data:
    def __init__(self, config):
        self.train_images_file = config['train_image_path']
        self.train_labels_file = config['train_label_path']
        self.test_images_file = config['test_image_path']
        self.test_labels_file = config['test_label_path']
        self.train_batch=config['train_batch']
        self.val_batch=config['val_batch']
        self.test_batch=config['test_batch']
    def load_train_dev(self):
        train_dev_set = MNISTDataset(self.train_images_file,self.train_labels_file)
        dataset_size = len(train_dev_set)
        train_size = int(0.9 * dataset_size)  # You can adjust the split ratio as needed
        dev_size = dataset_size - train_size
        train_set, val_set = random_split(train_dev_set, [train_size, dev_size])
        
        train_loader = DataLoader(train_set, batch_size=self.train_batch, num_workers=2,shuffle=True)
        val_loader = DataLoader(val_set, batch_size=self.val_batch, num_workers=2,shuffle=True)
        return train_loader,val_loader
    
    def load_test(self):
        test_set = MNISTDataset(self.test_images_file,self.test_labels_file)
        test_loader = DataLoader(test_set, batch_size=self.test_batch,num_workers=2,shuffle=False)
        return test_loader