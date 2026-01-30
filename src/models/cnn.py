from torch import nn
from torchvision import models


class GoogleNetFT(nn.Module):
    def __init__(self, num_classes: int, requires_grad: bool = False):
        super().__init__()
        self.model = models.googlenet(weights='IMAGENET1K_V1')
        for param in self.model.parameters():
            param.requires_grad = requires_grad

        # Changing the output layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        logits = self.model(x)
        return logits
