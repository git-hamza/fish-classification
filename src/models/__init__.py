import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

import constants


def resnet18(
    weights: ResNet18_Weights | None = None, output_labels: int = 9
) -> nn.Module:
    model_ft = models.resnet18(weights=weights)
    num_ftrs = model_ft.fc.in_features

    model_ft.fc = nn.Linear(num_ftrs, output_labels)
    model_ft = model_ft.to(constants.DEVICE)
    return model_ft
