import torch
from torch.utils.data import Dataset
from utils.feature_extraction import extract_features

class SERDataset(Dataset):
    def __init__(self, dataframe, scaler=None):
        self.filepaths = dataframe['filepath'].values
        self.labels = dataframe['label'].values
        self.scaler = scaler

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        label = self.labels[idx]

        features = extract_features(filepath)

        if features is None:
            return torch.zeros((173, 122), dtype=torch.float32), torch.tensor(0, dtype=torch.long)

        if self.scaler:
            features = self.scaler.transform(features)

        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        return features_tensor, torch.tensor(label, dtype=torch.long)

