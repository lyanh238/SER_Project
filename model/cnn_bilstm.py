import torch
import torch.nn as nn

class CNNBiLSTM(nn.Module):
    def __init__(self, num_classes, total_features, max_pad_length):
        super(CNNBiLSTM, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=(2, 2))
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=(2, 2))
        )

        with torch.no_grad():
            dummy_input = torch.randn(1, 1, max_pad_length, total_features)
            dummy_output = self.conv2(self.conv1(dummy_input))
            self.lstm_input_size = dummy_output.shape[1] * dummy_output.shape[3]
            self.lstm_sequence_length = dummy_output.shape[2]

        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(x.size(0), x.size(1), -1)

        lstm_out, _ = self.lstm(x)
        x = torch.mean(lstm_out, dim=1)

        logits = self.fc(x)
        return logits
