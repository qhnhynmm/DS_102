    
import os
import torch
from model.MLP import MLP_Model
from data_utils.load_data import Load_data
from evaluate.evaluate import compute_score
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
class Inference:
    def __init__(self,config):
        self.save_path=config['save_path']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.base_model = MLP_Model(config).to(self.device)
        self.dataloader = Load_data(config)
    def predict(self):
        test_data = self.dataloader.load_test()
        if os.path.exists(os.path.join(self.save_path, 'best_model.pth')):
            checkpoint = torch.load(os.path.join(self.save_path, 'best_model.pth'), map_location=self.device)
            self.base_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print('chưa train model mà đòi test hả')
        self.base_model.eval()
        true_labels = []
        pred_labels = []
        with torch.no_grad():
            for it,item in enumerate(tqdm(test_data)):
                images, labels = item['image'].to(self.device), item['label'].to(self.device)
                logits = self.base_model(images)
                true_labels.extend(labels.cpu().numpy())
                pred_labels.extend(logits.argmax(-1).cpu().numpy())
        test_acc,test_f1,test_precision,test_recall=compute_score(true_labels,pred_labels)
        print(f"test acc: {test_acc:.4f} | test f1: {test_f1:.4f} | test precision: {test_precision:.4f} | test recall: {test_recall:.4f}")
        print("confusion matrix:\n")
        print(confusion_matrix(true_labels,pred_labels))
        print('score trên từng nhãn:\n')
        print(classification_report(true_labels,pred_labels))
        
