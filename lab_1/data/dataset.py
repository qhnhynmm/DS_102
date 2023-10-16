import struct
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class MNISTDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class MNISTLoader:
    def __init__(self, config, transform=None):
        self.train_images_file = config['data']['train_image_path']
        self.train_labels_file = config['data']['train_label_path']
        self.data_images, self.data_labels = self.load_data(self.train_images_file,self.train_labels_file)
        self.transform = transform
        
        total_train_samples = int(0.8 * len(self.data_labels))
        total_dev_samples = int(0.9 * len(self.data_labels))
        total_test_samples = int(len(self.data_labels))
        self.train_dataset = MNISTDataset(self.data_images[:total_train_samples], self.data_labels[:total_train_samples], transform=self.transform)
        self.dev_dataset = MNISTDataset(self.data_images[total_train_samples:total_dev_samples], self.data_labels[total_train_samples:total_dev_samples], transform=self.transform)
        self.test_dataset = MNISTDataset(self.data_images[total_dev_samples:total_test_samples], self.data_labels[total_dev_samples:total_test_samples], transform=self.transform)

    def load_data(self, images_file, labels_file):
        with open(images_file, 'rb') as f:
            magic, num_images, num_rows, num_cols = struct.unpack('>IIII', f.read(16))
            images = np.fromfile(f, dtype=np.uint8).reshape(num_images, num_rows, num_cols)

        with open(labels_file, 'rb') as f:
            magic, num_items = struct.unpack('>II', f.read(8))
            labels = np.fromfile(f, dtype=np.uint8)

        return images, labels

    def get_train_loader(self, batch_size, shuffle=True):
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=shuffle)
        return train_loader

    def get_dev_loader(self, batch_size, shuffle=False):
        dev_loader = DataLoader(self.dev_dataset, batch_size=batch_size, shuffle=shuffle)
        return dev_loader
    
    def get_test_loader(self, batch_size, shuffle=False):
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=shuffle)
        return test_loader