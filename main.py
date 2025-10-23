import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.optim import lr_scheduler
from torchvision.models import ResNet18_Weights

import constants
from configs.config import CONFIG
from src.dataset.loading_data import DATA_TRANSFORM, DataProcessor
from src.eval import evaluate_model
from src.models import resnet18
from src.train import train_model
from src.utils import read_class_txt
from src.utils.logger import setup_logger

logger = setup_logger(constants.LOGGER_NAME, os.path.join(constants.LOG_DIR, "app.log"))


def training_pipeline() -> nn.Module:
    """
    Runs a complete training pipeline, executing training loop and evluating the model
    """
    data_processor = DataProcessor()
    image_datasets = data_processor.get_customdataset()

    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x],
            batch_size=CONFIG.batch_size,
            shuffle=True if x == "train" else False,
            num_workers=4,
        )
        for x in image_datasets
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in image_datasets}
    class_names = image_datasets["train"].classes

    model_ft = resnet18(ResNet18_Weights.IMAGENET1K_V1, len(class_names))

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=CONFIG.lr, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(
        model_ft,
        dataloaders,
        dataset_sizes,
        criterion,
        optimizer_ft,
        exp_lr_scheduler,
        CONFIG.epochs,
    )

    evaluate_model(model_ft, dataloaders["test"])
    return model_ft


def get_model_prediction(
    model: nn.Module, img_path: str, class_names: list[str]
) -> str:
    """
    Get inference on a single image
    """
    model.eval()

    img = Image.open(img_path).convert("RGB")
    img_tensor = DATA_TRANSFORM["test"](img).unsqueeze(0).to(constants.DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(probs, 1)
        pred_idx = int(preds[0].item())

    return class_names[pred_idx]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Running Pipeline")
    parser.add_argument("--mode", type=str, default="train")  # mode is train/test
    parser.add_argument(
        "--img_path", type=str, default="data/split_data/test/Forell/00013.png"
    )
    parser.add_argument("--ckpt_file", type=str, default="")
    args = parser.parse_args()

    if args.mode == "train":
        training_pipeline()
    else:
        class_names = read_class_txt(constants.CLASS_FILE)
        model_ft = resnet18(output_labels=len(class_names))
        best_model_params_path = args.ckpt_file or os.path.join(
            constants.CKPT_PATH, "ckpt.pt"
        )

        if os.path.isfile:
            state = torch.load(
                best_model_params_path, weights_only=True, map_location=constants.DEVICE
            )
            model_ft.load_state_dict(state)

            pred_label = get_model_prediction(model_ft, args.img_path, class_names)
            logger.info(f"Prediction: {pred_label}")
        else:
            logger.error(
                "model checkpoint does not exist. please provide a checkpoitn or train the model first"
            )
