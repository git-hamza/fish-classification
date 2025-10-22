import logging

import torch
from sklearn.metrics import accuracy_score

import constants

logger = logging.getLogger(constants.LOGGER_NAME)


def evaluate_model(model, test_loader):
    # Initialize dictionaries to store correct and total predictions
    correct_pred = {classname: 0 for classname in test_loader.dataset.classes}
    total_pred = {classname: 0 for classname in test_loader.dataset.classes}

    # Set the model to evaluation mode
    model.eval()

    # Track the ground truth labels and predictions
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            # Move the inputs and labels to the device
            inputs = inputs.to(constants.DEVICE)
            labels = labels.to(constants.DEVICE)

            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # Collect predictions and labels for metric calculations
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            # Update the correct and total predictions
            for label, prediction in zip(labels, preds):
                classname = test_loader.dataset.classes[label]
                if label == prediction:
                    correct_pred[classname] += 1
                total_pred[classname] += 1

    # Calculate accuracy per class
    accuracy_per_class = {
        classname: (
            correct_pred[classname] / total_pred[classname]
            if total_pred[classname] > 0
            else 0
        )
        for classname in test_loader.dataset.classes
    }

    # Calculate overall accuracy
    overall_accuracy = accuracy_score(all_labels, all_preds)

    # Print the evaluation results
    logger.info("Accuracy per class:")
    for classname, accuracy in accuracy_per_class.items():
        logger.info(f"{classname}: {accuracy:.4f}")

    logger.info(f"Overall Accuracy: {overall_accuracy:.4f}")
    logger.info(f"Overall Accuracy: {overall_accuracy:.4f}")
