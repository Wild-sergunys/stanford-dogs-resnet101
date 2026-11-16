"""ResNet-101 с классификацией 120 пород собак."""
from torch import nn
from torchvision.models import ResNet101_Weights, resnet101


def create_model(pretrained=True):
    weights = ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet101(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 120)
    return model
