import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.optim import lr_scheduler

import constants
from src.dataset.loading_data import DATA_TRANSFORM, DataProcessor
from src.eval import evaluate_model
from src.models import pretrained
from src.train import train_model
from src.utils.func_util import read_class_txt
from src.utils.logger import setup_logger

logger = setup_logger(constants.LOGGER_NAME, os.path.join(constants.LOG_DIR, "app.log"))


def training_pipeline(
    num_epochs=2,
    lr=0.001,
):
    data_processor = DataProcessor()
    image_datasets = data_processor.get_dataset()
    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x],
            batch_size=4,
            shuffle=True if x == "train" else False,
            num_workers=4,
        )
        for x in image_datasets
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in image_datasets}
    class_names = image_datasets["train"].classes

    model_ft = pretrained.resnet18("IMAGENET1K_V1", len(class_names))

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=lr, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(
        model_ft,
        dataloaders,
        dataset_sizes,
        criterion,
        optimizer_ft,
        exp_lr_scheduler,
        num_epochs,
    )

    evaluate_model(model_ft, dataloaders["test"])


def get_model_prediction(model, img_path, class_names):
    model.eval()

    img = Image.open(img_path)
    img = DATA_TRANSFORM["test"](img)
    img = img.unsqueeze(0)
    img = img.to()

    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)

    return class_names[preds[0]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Running Pipeline")
    parser.add_argument("--mode", type=str, default="test")  # mode is train/test
    parser.add_argument(
        "--img_path", type=str, default="data/split_data/test/Forell/00013.png"
    )
    args = parser.parse_args()

    if args.mode == "train":
        training_pipeline()
    else:
        class_names = read_class_txt(constants.CLASS_FILE)
        model_ft = pretrained.resnet18(output_labels=len(class_names))
        best_model_params_path = os.path.join(constants.CKPT_PATH, "ckpt.pt")
        model_ft.load_state_dict(torch.load(best_model_params_path, weights_only=True))

        prediction = get_model_prediction(model_ft, args.img_path, class_names)
