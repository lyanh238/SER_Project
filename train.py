import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
from model.cnn_bilstm import CNNBiLSTM
from utils.data_loader import SERDataset
from utils.evaluation import calculate_metrics
from utils.visualization import plot_training

# Load your dataframe (replace with actual loading code)
df = pd.read_csv('your_dataframe.csv')
label_to_int = {label: idx for idx, label in enumerate(df['emotion'].unique())}

X = df['filepath']
y = df['label']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

train_df = pd.DataFrame({'filepath': X_train, 'label': y_train})
val_df = pd.DataFrame({'filepath': X_val, 'label': y_val})

scaler = StandardScaler()
train_dataset = SERDataset(train_df, scaler)
val_dataset = SERDataset(val_df, scaler)

BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNNBiLSTM(len(label_to_int), 122, 173).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)

best_val_f1 = 0
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'train_f1': [], 'val_f1': []}

for epoch in range(30):
    model.train()
    total_loss = 0
    y_true, y_pred = [], []

    for features, labels in tqdm(train_loader):
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

    train_acc, train_f1 = calculate_metrics(y_true, y_pred)
    avg_train_loss = total_loss / len(train_loader)

    model.eval()
    total_val_loss = 0
    y_val_true, y_val_pred = [], []

    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            y_val_true.extend(labels.cpu().numpy())
            y_val_pred.extend(predicted.cpu().numpy())

    val_acc, val_f1 = calculate_metrics(y_val_true, y_val_pred)
    avg_val_loss = total_val_loss / len(val_loader)

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), 'models/best_model.pt')

    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    history['train_f1'].append(train_f1)
    history['val_f1'].append(val_f1)

    scheduler.step(val_f1)

    print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")

plot_training(history)