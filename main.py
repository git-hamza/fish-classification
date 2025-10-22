
from dataset import loading_data
from train import train_model
from eval import evaluate_model
from dataset.loading_data import DataProcessor

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from models import pretrained

def training_pipeline():
    data_processor = DataProcessor()
    image_datasets = data_processor.get_dataset()
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                shuffle=True if x=="train" else False, num_workers=4)
                for x in image_datasets}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in image_datasets}
    class_names = image_datasets['train'].classes

    model_ft = pretrained.resnet18('IMAGENET1K_V1', len(class_names))

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, dataloaders, dataset_sizes, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=2)
    
    evaluate_model(model_ft, dataloaders["test"])

def inference(model, img):
    pass


if __name__ == "__main__":
    training_pipeline()