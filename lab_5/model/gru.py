from typing import List, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from text_module.embedding import Embedding


class GRU_Model(nn.Module):
    def __init__(self,config: Dict, num_labels: int):
     
        super(GRU_Model, self).__init__()
        self.num_labels = num_labels
        self.intermediate_dims = config["model"]["intermediate_dims"]
        self.dropout=config["model"]["dropout"]
        self.text_embbeding = Embedding(config)
        self.gru = nn.GRU(self.intermediate_dims, self.intermediate_dims,
                          num_layers=config['model']['num_layer'],dropout=self.dropout)
        self.classifier = nn.Linear(self.intermediate_dims,self.num_labels)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, text: List[str], labels: Optional[torch.LongTensor] = None):
        embbed = self.text_embbeding(text)
        gru_feat, _ = self.gru(embbed)
        mean_pooling = torch.mean(gru_feat, dim=1)
        logits = self.classifier(mean_pooling)
        logits = F.log_softmax(logits, dim=-1)
        out = {
            "logits": logits
        }
        if labels is not None:
            loss = self.criterion(logits, labels)
            out["loss"] = loss
        
        return out

def createGRU_Model(config: Dict, answer_space: List[str]) -> GRU_Model:
    return GRU_Model(config, num_labels=len(answer_space))