import torch
import torch.nn as nn
import torch.nn.functional as F


class Spectral1DCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=4,
            out_channels=16,
            kernel_size=11,
            padding=5
        )
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(
            in_channels=16,
            out_channels=32,
            kernel_size=9,
            padding=4
        )
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(0.20)

        self.conv3 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=7,
            padding=3
        )
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        self.conv4 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=5,
            padding=2
        )
        self.bn4 = nn.BatchNorm1d(128)
        self.pool4 = nn.MaxPool1d(kernel_size=2)

        self.conv5 = nn.Conv1d(
            in_channels=128,
            out_channels=264,
            kernel_size=3,
            padding=1
        )
        self.bn5 = nn.BatchNorm1d(264)

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(264, 128)
        self.dropout1 = nn.Dropout(0.35)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x shape: (batch_size, channels, sequence_length)

        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = F.relu(self.bn5(self.conv5(x)))

        x = self.global_pool(x)

        # shape becomes (batch_size, 128, 1)
        x = x.squeeze(-1)

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)

        return x