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