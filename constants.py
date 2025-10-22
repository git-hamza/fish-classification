import os
import torch

LOGGER_NAME = "app"
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
RAWDATA_DIR = os.path.join(DATA_DIR, "raw_data")
SPLITDATA_DIR = os.path.join(DATA_DIR, "split_data")
CLASS_FILE = os.path.join(BASE_DIR, "dataset/classes.txt")
CKPT_PATH = os.path.join(BASE_DIR, "checkpoint")

# We want to be able to train our model on an `accelerator <https://pytorch.org/docs/stable/torch.html#accelerators>`__
# such as CUDA, MPS, MTIA, or XPU. If the current accelerator is available, we will use it. Otherwise, we use the CPU.
DEVICE = "cpu"
