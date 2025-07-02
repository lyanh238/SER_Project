from sklearn.metrics import accuracy_score, f1_score

def calculate_metrics(y_true, y_pred, average='macro'):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average=average)
    return accuracy, f1