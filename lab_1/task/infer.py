    
import os
import torch
from model.MLP import MLP_Model
from data.dataset import MNISTLoader
from evaluate.evaluate import compute_score
class Inference:
    def __init__(self,config):
        self.base_model = MLP_Model(config).to(self.device)
        self.dataloader = Load_data(config)
        self.save_path=config.save_path
    def predict(self):
        test_data = self.dataloader.load_test()
        if os.path.exists(os.path.join(self.save_path, 'best_model.pth')):
            checkpoint = torch.load(os.path.join(self.save_path, 'best_model.pth'), map_location=self.device)
            self.base_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print('chưa train model mà đòi test hả')
        self.base_model.eval()

        test_acc = 0.
        true_labels = []
        pred_labels = []
        with torch.no_grad():
            for images, labels in test_data:
                images, labels = images.to(self.device), labels.to(self.device)
                logits = self.base_model(images)
                true_labels.extend(labels.cpu().numpy())
                pred_labels.extend(logits.argmax(1).cpu().numpy())
        test_acc,test_f1,test_precision,test_recall=compute_score(true_labels,pred_labels)
        print(f"test acc: {test_acc:.4f} | test f1: {test_f1:.4f} | test precision: {test_precision:.4f} | test recall: {test_recall:.4f}")
        
