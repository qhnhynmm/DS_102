from typing import List, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from text_module.embedding import Embedding

class LSTM_Model(nn.Module):
    def __init__(self,config: Dict, num_labels: int):
     
        super(LSTM_Model, self).__init__()
        self.num_labels = num_labels
        self.intermediate_dims = config["model"]["intermediate_dims"]
        self.dropout=config["model"]["dropout"]
        self.text_embbeding = Embedding(config)
        self.lstm = nn.LSTM(self.intermediate_dims, self.intermediate_dims,
                          num_layers=config['model']['num_layer'],dropout=self.dropout)
        self.classifier = nn.Linear(self.intermediate_dims,self.num_labels)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, text: List[str], labels: Optional[torch.LongTensor] = None):
        embbed = self.text_embbeding(text)
        lstm_feat, _ = self.lstm(embbed)
        mean_pooling = torch.mean(lstm_feat, dim=1)
        logits = self.classifier(mean_pooling)
        logits = F.log_softmax(logits, dim=-1)
        out = {
            "logits": logits
        }
        if labels is not None:
            loss = self.criterion(logits, labels)
            out["loss"] = loss
        
        return out

def createLSTM_Model(config: Dict, answer_space: List[str]) -> LSTM_Model:
    return LSTM_Model(config, num_labels=len(answer_space))