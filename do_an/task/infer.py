from typing import List, Dict, Optional,Text
from data_utils.load_data import Get_Loader
import torch
import os
import pandas as pd
from tqdm import tqdm
from eval_metric.evaluate import ScoreCalculator
from model.build_model import build_model
class Predict:
    def __init__(self, config: Dict):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_path=os.path.join(config["train"]["output_dir"], "best_model.pth")
        self.model = build_model(config)
        self.dataloader = Get_Loader(config)
        self.compute_score = ScoreCalculator()
    def predict_submission(self):
        # Load the model
        print("Loading the best model...")
        checkpoint = torch.load(self.checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)

        # Obtain the prediction from the model
        print("Obtaining predictions...")
        test =self.dataloader.load_test()
        submits = []
        ids = []
        gts = []
        self.model.eval()
        with torch.no_grad():
            for it, (sents, labels, id) in enumerate(tqdm(test)):
                with torch.autocast(device_type='cuda', dtype=torch.float32, enabled=True):
                    logits = self.model(sents)
                    preds = torch.round(logits)

                submits.append(preds)
                gts.append(labels)
                if isinstance(id, torch.Tensor):
                    ids.extend(id.tolist())
                else:
                    ids.extend(id)
        submits = torch.cat(submits, dim=0).cpu()
        gts = torch.cat(gts, dim=0).cpu()

        print('accuracy on test:', self.compute_score.acc(gts, submits))
        print('f1 on test:', self.compute_score.f1(gts, submits))
        print('recall on test:', self.compute_score.recall(gts, submits))
        print('precision on test:', self.compute_score.precision(gts, submits))
        print('auc on test:', self.compute_score.auc(gts, submits))

        data = {'id': ids, 'generated': submits.numpy()}  # Move submits to CPU and convert to NumPy array
        df = pd.DataFrame(data)
        df.to_csv('./submission.csv', index=False)
