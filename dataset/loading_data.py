import requests
import constants
import os
from utils.func_util import get_classes, url_format
import splitfolders
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory
from dataset.custom_dataset import CustomDataset
from utils.func_util import read_class_txt

logger = logging.getLogger(constants.LOGGER_NAME)

class DataProcessor:
    def __init__(self, resize=(224,224)):
        self._data_url = "https://stnordaxoninternal.blob.core.windows.net"
        self._image_count = 1000
        self._datasetname = "fish-dataset"
        self.classes = read_class_txt(constants.CLASS_FILE)

        self.transform = {
            'train': v2.Compose([
                v2.Resize(resize),
                v2.ToTensor(),
                v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': v2.Compose([
                v2.Resize(resize),
                v2.ToTensor(),
                v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': v2.Compose([
                v2.Resize(resize),
                v2.ToTensor(),
                v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }

    def _url_format(self, class_id, img_name):
        return f"{self._data_url}/{self._datasetname}/{class_id}/{img_name}"

    def get_data(self):
        for class_name in self.classes:
            class_dir = os.path.join(constants.DATA_DIR, class_name)
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)

            for i in range(1, self._image_count+1):
                image_name = f"{i:05d}.png"
                url_ = url_format(self._data_url , self._datasetname, class_name, image_name)
                file_path = os.path.join(class_dir, image_name)
                if not os.path.isfile(file_path):
                    response = requests.get(url_)
                    if response.status_code == 200:
                        with open(file_path, 'wb') as file:
                            file.write(response.content)
                    else:
                        logging.warning('Failed to download file')
            else:
                logger.info(f"all file for {class_name} downloaded")


    def split_data(self):
        if not os.path.exists(constants.SPLITDATA_DIR):
            if not os.path.exists(constants.RAWDATA_DIR):
                self.get_data()
        
            splitfolders.ratio(constants.RAWDATA_DIR, output=constants.SPLITDATA_DIR, seed=1337, ratio=(.8, 0.1,0.1))

        data_dict = {
            "train": os.path.join(constants.SPLITDATA_DIR, "train"),
            "val": os.path.join(constants.SPLITDATA_DIR, "val"),
            "test": os.path.join(constants.SPLITDATA_DIR, "test")
        }
        return data_dict

    def get_dataset(self):
        datadir_dict = self.split_data()
        image_datasets = {x: datasets.ImageFolder(self.datadir_dict[x],
                                                DATA_TRANSFORM[x])
                        for x in self.datadir_dict}
        
        return image_datasets

    def get_customdataset(self):
        datadir_dict = self.split_data()
        image_datasets = {x: CustomDataset(self.datadir_dict[x],
                                                DATA_TRANSFORM[x])
                        for x in self.datadir_dict}
        
        return image_datasets

