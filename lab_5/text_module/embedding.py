import torch
import torch.nn as nn
from data_utils.vocab import Vocab
from typing import List
class Embedding(nn.Module):
    def __init__(self, config):
        super(Embedding, self).__init__()
        self.embedding_dim = config['text_embedding']['d_features']
        self.vocab = Vocab(config)
        self.embedding = nn.Embedding(self.vocab.vocab_size(), self.embedding_dim)
        self.dropout = nn.Dropout(config['text_embedding']['dropout'])
        self.gelu = nn.GELU()
        self.max_length = config["tokenizer"]["max_length"]
        self.proj = nn.Linear(self.embedding_dim, config["text_embedding"]["d_model"])
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    def pad_list(self, list: List, max_len: int, value):
        pad_value_list = [value] * (max_len - len(list))
        list.extend(pad_value_list)
        return list
    
    def forward(self, input_texts):
        X=[]
        for s in input_texts:
            sen=[self.vocab.word_to_idx.get('[CLS]')]
            for w in s.split():
                sen.append(self.vocab.word_to_idx.get(w,self.vocab.word_to_idx['[UNK]']))
            sen=sen[:self.max_length-1]
            sen.append(self.vocab.word_to_idx.get('[SEP]'))
            sen=self.pad_list(sen,self.max_length, self.vocab.pad_token_id())
            X.append(torch.tensor(sen,dtype=torch.int))
        out = self.embedding(torch.stack(X).to(self.device))
        out = self.proj(out)
        out = self.dropout(self.gelu(out))
        return out