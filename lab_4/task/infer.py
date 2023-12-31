    
import os
import torch
from model.resnet50 import ResNet50
from model.vgg19 import VGG19
from data_utils.load_data import Load_data
from evaluate.evaluate import compute_score
from tqdm import tqdm
class Inference:
    def __init__(self,config):
        self.save_path=os.path.join(config['save_path'],config['model'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = config['model']
        if self.model == 'vgg19':
            self.base_model = VGG19(config).to(self.device)
        if self.model == 'resnet50':
            self.base_model = ResNet50(config).to(self.device)
        self.dataloader = Load_data(config)
    def predict(self):
        test_data = self.dataloader.load_test()
        if os.path.exists(os.path.join(self.save_path, 'best_model.pth')):
            checkpoint = torch.load(os.path.join(self.save_path, 'best_model.pth'), map_location=self.device)
            self.base_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print('train đi')
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
        