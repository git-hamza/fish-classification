import os

import torch

LOGGER_NAME = "app"
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
RAWDATA_DIR = os.path.join(DATA_DIR, "raw_data")
SPLITDATA_DIR = os.path.join(DATA_DIR, "split_data")
CLASS_FILE = os.path.join(BASE_DIR, "src/dataset/classes.txt")
CKPT_PATH = os.path.join(BASE_DIR, "checkpoint")
CONFIG_FILE_PATH = os.path.join(BASE_DIR, "config.yaml")
LOG_DIR = os.path.join(BASE_DIR, "logs")

for dir in [LOG_DIR, CKPT_PATH]:
    os.makedirs(LOG_DIR, exist_ok=True)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
