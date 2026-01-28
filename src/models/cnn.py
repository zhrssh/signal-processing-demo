
from torch import nn


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 64, 3),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 32, 3),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 16, 3),
            nn.AdaptiveAvgPool2d((3, 3)),
            nn.Flatten()
        )

        self.fc_stack = nn.Sequential(
            nn.Linear(16 * 3 * 3, 120),
            nn.ReLU(),
            nn.Linear(120, 32),
            nn.ReLU(),
            nn.Linear(32, 5)
        )

    def forward(self, x):
        return self.fc_stack(self.conv_stack(x))
