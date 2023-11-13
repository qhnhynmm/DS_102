from typing import List, Dict, Optional
import torch
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self, config):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=config['image_C'], out_channels=6, kernel_size = 5, padding = 2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=5, out_channels = 16 , kernel_size=16)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, config['num_classes'])

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        x = torch.softmax(x, dim=-1)
        return x

class LeNet_Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lenet = LeNet(config)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, imgs, labels=None):
        if labels is not None:
            logits = self.lenet(imgs)
            loss = self.loss_fn(logits, labels)
            return logits, loss
        else:
            logits = self.lenet(imgs)
            return logits
