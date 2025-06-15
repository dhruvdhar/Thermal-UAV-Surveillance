# ThermalCNN Model
import torch.nn as nn

class ThermalCNN(nn.Module):
    def __init__(self, dropout=0.3, filters=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, filters, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(filters, filters*2, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(filters*2*32*32, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 4)
        )
    def forward(self, x):
        return self.fc(self.conv(x))
