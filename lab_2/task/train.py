import torch
import torch.nn as nn
import torch.optim as optim
import os
import requests
from PIL import Image
from io import BytesIO
from data_utils.load_data import Load_data
from evaluate.evaluate import compute_score
from model.MLP import MLP_Model
from tqdm import tqdm
from utils.regularzation import l2_regularzation,l1_regularzation
from utils.init_method import ones_init,xavier_init

class Classify_Task:
    def __init__(self, config):
        self.num_epochs = config['num_epochs']
        self.patience = config['patience']
        self.learning_rate = config['learning_rate']
        self.best_metric=config['best_metric']
        self.save_path=config['save_path']
        self.lambda_=config['lambda']
        self.momentum=config['momentum']
        self.regularzation_type=config['regularzation_type']
        self.dataloader = Load_data(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.base_model = MLP_Model(config).to(self.device)
        if config['init_type']=='ones':
            ones_init(self.base_model)
        if config['init_type']=='xavier':
            xavier_init(self.base_model)
        if config['optimizer_type']=='adam':
            self.optimizer = optim.Adam(self.base_model.parameters(), lr=self.learning_rate)
        elif config['optimizer_type']=='sgd':
            self.optimizer = optim.SGD(lr=self.learning_rate,momentum=self.momentum)
        elif config['optimizer_type']=='rmsprop':
            self.optimizer = optim.RMSprop(lr=self.learning_rate,momentum=self.momentum)
        else:
            self.optimizer = optim.Adam(self.base_model.parameters(), lr=self.learning_rate)
    def training(self):
        if not os.path.exists(self.save_path):
          os.makedirs(self.save_path)

        train,valid = self.dataloader.load_train_dev()

        if os.path.exists(os.path.join(self.save_path, 'last_model.pth')):
            checkpoint = torch.load(os.path.join(self.save_path, 'last_model.pth'))
            self.base_model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print('loaded the last saved model!!!')
            initial_epoch = checkpoint['epoch'] + 1
            print(f"continue training from epoch {initial_epoch}")
        else:
            initial_epoch = 0
            print("first time training!!!")
            train_loss = 0.

        if os.path.exists(os.path.join(self.save_path, 'best_model.pth')):
            checkpoint = torch.load(os.path.join(self.save_path, 'best_model.pth'))
            best_score = checkpoint['score']
        else:
            best_score = 0.
            
        threshold=0
        self.base_model.train()
        for epoch in range(initial_epoch, self.num_epochs + initial_epoch):
            valid_acc = 0.
            valid_f1=0.
            valid_precision=0.
            valid_recall=0.
            train_loss = 0.
            for it,item in enumerate(tqdm(train)):
                images, labels = item['image'].to(self.device), item['label'].to(self.device)
                self.optimizer.zero_grad()
                logits,loss = self.base_model(images,labels)
                if self.regularzation_type=='l2':
                    loss = loss + self.lambda_*l2_regularzation(self.base_model)
                if self.regularzation_type=='l1':
                    loss = loss + self.lambda_*l1_regularzation(self.base_model)
                loss.backward()
                self.optimizer.step()
                train_loss += loss

            with torch.no_grad():
                for it,item in enumerate(tqdm(valid)):
                    images, labels = item['image'].to(self.device), item['label'].to(self.device)
                    logits = self.base_model(images)
                    preds = logits.argmax(-1)
                    acc, f1, precision, recall=compute_score(labels.cpu().numpy(),preds.cpu().numpy())
                    valid_acc+=acc
                    valid_f1+=f1
                    valid_precision+=precision
                    valid_recall+=recall
        
            train_loss /= len(train)
            valid_acc /= len(valid)
            valid_f1 /= len(valid)
            valid_precision /= len(valid)
            valid_recall /= len(valid)


            print(f"epoch {epoch + 1}/{self.num_epochs + initial_epoch}")
            print(f"train loss: {train_loss:.4f}")
            print(f"valid acc: {valid_acc:.4f} | valid f1: {valid_f1:.4f} | valid precision: {valid_precision:.4f} | valid recall: {valid_recall:.4f}")

            if self.best_metric =='accuracy':
                score=valid_acc
            if self.best_metric=='f1':
                score=valid_f1
            if self.best_metric=='precision':
                score=valid_precision
            if self.best_metric=='recall':
                score=valid_recall
            # save the last model
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.base_model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'score': score}, os.path.join(self.save_path, 'last_model.pth'))
            
            # save the best model
            if epoch > 0 and score < best_score:
              threshold += 1
            else:
              threshold = 0

            if score > best_score:
                best_score = score
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.base_model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'score':score}, os.path.join(self.save_path, 'best_model.pth'))
                print(f"saved the best model with {self.best_metric} of {score:.4f}")
            
            # early stopping
            if threshold >= self.patience:
                print(f"early stopping after epoch {epoch + 1}")
                break