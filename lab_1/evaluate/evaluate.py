from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

def compute_score(labels,preds):
    acc=accuracy_score(labels,preds)
    f1=f1_score(labels,preds,average='macro')
    precision=precision_score(labels,preds)
    recall=recall_score(labels,preds)
    return acc,f1,precision,recall