import logging

import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

import constants

logger = logging.getLogger(constants.LOGGER_NAME)


def evaluate_model(model: torch.nn.Module, test_loader: DataLoader) -> None:
    """
    model evaluation
    """
    correct_pred = {classname: 0 for classname in test_loader.dataset.classes}
    total_pred = {classname: 0 for classname in test_loader.dataset.classes}

    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(constants.DEVICE)
            labels = labels.to(constants.DEVICE)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

            for label_tensor, pred_tensor in zip(labels, preds):
                label_idx = int(label_tensor.item())
                pred_idx = int(pred_tensor.item())
                classname = test_loader.dataset.classes[label_idx]
                if label_idx == pred_idx:
                    correct_pred[classname] += 1
                total_pred[classname] += 1

    accuracy_per_class = {
        classname: (
            correct_pred[classname] / total_pred[classname]
            if total_pred[classname] > 0
            else 0.0
        )
        for classname in test_loader.dataset.classes
    }

    overall_accuracy = accuracy_score(all_labels, all_preds)

    logger.info("Accuracy per class:")
    for classname, accuracy in accuracy_per_class.items():
        logger.info(f"{classname}: {accuracy:.4f}")

    logger.info(f"Overall Accuracy: {overall_accuracy:.4f}")
