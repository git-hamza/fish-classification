import logging
import os
import time
from typing import Dict

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

import constants

logger = logging.getLogger(constants.LOGGER_NAME)


def train_model(
    model: torch.nn.Module,
    dataloaders: Dict[str, DataLoader],
    dataset_sizes: Dict[str, int],
    criterion: torch.nn.Module,
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    num_epochs: int = 25,
) -> torch.nn.Module:
    """
    training loop
    """
    since = time.time()

    best_acc = 0.0
    best_model_params_path = None

    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch}/{num_epochs - 1}")
        logger.info("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss: float = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(constants.DEVICE)
                labels = labels.to(constants.DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += float(loss.item()) * inputs.size(0)
                running_corrects += int(torch.sum(preds == labels).item())

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            logger.info(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_params_path = os.path.join(constants.CKPT_PATH, "ckpt.pt")
                torch.save(model.state_dict(), best_model_params_path)

    time_elapsed = time.time() - since
    logger.info(
        f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
    )
    logger.info(f"Best val Acc: {best_acc:.4f}")

    if best_model_params_path:
        state = torch.load(
            best_model_params_path, weights_only=True, map_location=constants.DEVICE
        )
        model.load_state_dict(state)

    return model
