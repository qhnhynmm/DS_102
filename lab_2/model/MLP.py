from typing import List, Dict, Optional
import torch
import torch.nn as nn

class MLP_1(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.image_W = config['image_W']
        self.image_H = config['image_H']
        self.image_C = config['image_C']
        self.num_classes = config['num_classes']
        self.fc=nn.Linear(self.image_C*self.image_H*self.image_W,self.num_classes)
    def forward(self, x):
        x = x.view(x.shape[0],x.shape[1]*x.shape[2])
        x = self.fc(x)
        x = torch.softmax(x,dim=-1)
        return x

class MLP_2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.image_W = config['image_W']
        self.image_H = config['image_H']
        self.image_C = config['image_C']
        self.num_classes = config['num_classes']
        self.fc1=nn.Linear(self.image_C*self.image_H*self.image_W,512)
        self.fc2=nn.Linear(512,256)
        self.fc3=nn.Linear(256,self.num_classes)
        self.relu=nn.ReLU()
    def forward(self, x):
        x = x.view(x.shape[0],x.shape[1]*x.shape[2])
        x = self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))
        x = torch.softmax(x,dim=-1)
        return x

class MLP_3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.image_W = config['image_W']
        self.image_H = config['image_H']
        self.image_C = config['image_C']
        self.num_classes = config['num_classes']
        self.fc1=nn.Linear(self.image_C*self.image_H*self.image_W,512)
        self.batch_n1=nn.BatchNorm1d(512)
        self.fc2=nn.Linear(512,256)
        self.batch_n2=nn.BatchNorm1d(256)
        self.fc3=nn.Linear(256,self.num_classes)
        self.relu=nn.ReLU()
    def forward(self, x):
        x = x.view(x.shape[0],x.shape[1]*x.shape[2])
        x = self.fc3(self.relu(self.batch_n2(self.fc2(self.relu(self.batch_n1(self.fc1(x)))))))
        x = torch.softmax(x,dim=-1)
        return x

class MLP_Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config['model']=='model_1':
            self.mlp=MLP_1(config)
        if config['model']=='model_2':
            self.mlp=MLP_2(config)
        if config['model']=='model_3':
            self.mlp=MLP_3(config)
        self.loss_fn=nn.CrossEntropyLoss()
    def forward(self,imgs,labels=None):
        if labels is not None:
            logits=self.mlp(imgs)
            loss = self.loss_fn(logits, labels)
            return logits,loss
        else:
            logits=self.mlp(imgs)
            return logits
            
