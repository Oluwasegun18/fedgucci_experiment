import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """Small CNN for CIFAR-10. No BatchNorm to simplify stateless connectivity loss."""
    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        h = self.features(x)
        h = h.flatten(1)
        return self.classifier(h)


def build_model(dataset: str, num_classes: int = 10):
    dataset = dataset.lower()
    if dataset == "cifar10":
        return SimpleCNN(in_channels=3, num_classes=num_classes)
    if dataset == "fashionmnist":
        return SimpleCNN(in_channels=1, num_classes=num_classes)
    raise ValueError(f"Unknown dataset: {dataset}")
