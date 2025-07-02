import torch
import pandas as pd
from model.cnn_bilstm import CNNBiLSTM
from utils.data_loader import SERDataset
from utils.evaluation import calculate_metrics
from utils.visualization import plot_confusion_matrix
from torch.utils.data import DataLoader

# Load your dataframe and mappings
df = pd.read_csv('your_dataframe.csv')
label_to_int = {label: idx for idx, label in enumerate(df['emotion'].unique())}
int_to_label = {v: k for k, v in label_to_int.items()}

X_test = df['filepath']
y_test = df['label']

test_df = pd.DataFrame({'filepath': X_test, 'label': y_test})

scaler = None  # Load your trained scaler if needed
test_dataset = SERDataset(test_df, scaler)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = CNNBiLSTM(len(label_to_int), 122, 173)
model.load_state_dict(torch.load('models/best_model.pt', map_location='cpu'))
model.eval()

y_true, y_pred = [], []

with torch.no_grad():
    for features, labels in test_loader:
        outputs = model(features)
        _, predicted = torch.max(outputs.data, 1)
        y_true.extend(labels.numpy())
        y_pred.extend(predicted.numpy())

accuracy, f1 = calculate_metrics(y_true, y_pred)
print(f"Test Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

plot_confusion_matrix(y_true, y_pred, list(label_to_int.keys()))