import torch.nn as nn
from torchvision import models

import constants


def resnet18(weights=None, output_labels=9):
    model_ft = models.resnet18(weights=weights)
    num_ftrs = model_ft.fc.in_features

    model_ft.fc = nn.Linear(num_ftrs, output_labels)
    model_ft = model_ft.to(constants.DEVICE)
    return model_ft
