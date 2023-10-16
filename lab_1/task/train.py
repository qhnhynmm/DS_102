import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.dataset import MNISTLoader
from model.MLP import MLP_Model
import torch.optim as optim
import os
class Classify_Task:
    def __init__(self, config):
        self.num_epochs = config['train']['num_epochs']
        self.patience = config['train']['patience']
        self.learning_rate = config['train']['learning_rate']
        self.best_metric=config['train']['best_metric']
        self.save_path=config['train']['save_path']
        self.dataloader = MNISTLoader(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.base_model = MLP_Model(config).to(self.device)
        self.optimizer = optim.Adam(self.base_model.parameters(), lr=self.learning_rate)
    def training(self):
        if not os.path.exists(self.save_path):
          os.makedirs(self.save_path)

        train = self.dataloader.get_train_loader(batch_size = 32)
        valid = self.dataloader.get_dev_loader(batch_size = 32)

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
            train_loss = 0.
            for images, labels in train:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                logits,loss = self.base_model(images,images)
                loss.backward()
                self.optimizer.step()
                train_loss += loss

            with torch.no_grad():
                for images, labels in valid:
                    images, labels = images.to(self.device), labels.to(self.device)
                    logits = self.base_model(images)
                    preds = logits.argmax(1)
                    valid_acc, valid_f1, valid_precision, valid_recall=compute_score(labels.cpu().numpy(),preds.cpu().numpy())

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