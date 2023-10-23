import torch
def l1_regularzation(model):
    l1=0
    for param in model.parameters():
        l1+=torch.norm(param, p=1)
    return l1 

def l2_regularzation(model):
    l2=0
    for param in model.parameters():
        l2+=torch.norm(param, p=2)
    return l2
