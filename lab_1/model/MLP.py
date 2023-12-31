from typing import List, Dict, Optional
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim=config['train']['hidden_dim']
        self.image_W = config['train']['image_W']
        self.image_H = config['train']['image_H']
        self.image_C = config['train']['image_C']
        self.num_classes = config['train']['num_classes']
        self.mlp=nn.Linear(self.image_C*self.image_H*self.image_W,self.hidden_dim)
        self.fc=nn.Linear(self.hidden_dim,self.num_classes)
    def forward(self, x):
        x=x.view(x.shape[0],x.shape[1]*x.shape[2])
        x = self.mlp(x)
        x = self.fc(x)
        x = torch.softmax(x, dim=1)
        return x

class MLP_Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp=MLP(config)
        self.loss_fn=nn.CrossEntropyLoss()
    def forward(self,imgs,labels=None):
        if labels is not None:
            logits=self.mlp(imgs)
            loss = self.loss_fn(logits, labels)
            return logits,loss
        else:
            logits=self.mlp(imgs)
            return logits

