from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, precision_score

class ScoreCalculator:
    def acc(self, labels, preds) -> float:
        preds = preds.cpu().numpy()
        labels = labels.cpu().numpy()
        return accuracy_score(labels, preds)
    
    def f1(self, labels, preds) -> float:
        preds = preds.cpu().numpy()
        labels = labels.cpu().numpy()
        return f1_score(labels, preds, average='macro')
    
    def auc(self, labels, logits) -> float:
        logits = logits.cpu().numpy()
        labels = labels.cpu().numpy()
        return roc_auc_score(labels, logits)
    
    def recall(self, labels, preds) -> float:
        preds = preds.cpu().numpy()
        labels = labels.cpu().numpy()
        return recall_score(labels, preds, average='macro')
    
    def precision(self, labels, preds) -> float:
        preds = preds.cpu().numpy()
        labels = labels.cpu().numpy()
        return precision_score(labels, preds, average='macro')
