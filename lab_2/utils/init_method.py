import torch.nn as nn
def ones_init(model):
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.ones_(module.weight)
            nn.init.ones_(module.bias)

def xavier_init(model):
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
