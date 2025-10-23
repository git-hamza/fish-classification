import logging
import os

import requests
import splitfolders
from torchvision.transforms import v2

import constants
from configs.config import CONFIG
from src.dataset.custom_dataset import CustomDataset
from src.utils import read_class_txt

logger = logging.getLogger(constants.LOGGER_NAME)

DATA_TRANSFORM = {
    "train": v2.Compose(
        [
            v2.Resize((224, 224)),
            v2.ToTensor(),
            v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    "val": v2.Compose(
        [
            v2.Resize((224, 224)),
            v2.ToTensor(),
            v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    "test": v2.Compose(
        [
            v2.Resize((224, 224)),
            v2.ToTensor(),
            v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}


class DataProcessor:
    def __init__(
        self,
    ):
        self.classes = read_class_txt(constants.CLASS_FILE)
        self.fetch_data()

    def _url_format(self, class_id, img_name):
        return f"{CONFIG.data_url}/{class_id}/{img_name}"

    def fetch_data(self):
        # TODO: fetching data need to be moved to async
        """
        fetching data from urls
        """
        if not os.path.exists(constants.RAWDATA_DIR):
            logger.info("raw data does not exist. Downloading the data")
            for class_name in self.classes:
                class_dir = os.path.join(constants.RAWDATA_DIR, class_name)
                if not os.path.exists(class_dir):
                    os.makedirs(class_dir)

                for i in range(1, CONFIG.image_count + 1):
                    image_name = f"{i:05d}.png"
                    url_ = self._url_format(class_name, image_name)
                    file_path = os.path.join(class_dir, image_name)
                    if not os.path.isfile(file_path):
                        response = requests.get(url_)
                        if response.status_code == 200:
                            with open(file_path, "wb") as file:
                                file.write(response.content)
                        else:
                            logging.warning("Failed to download file")
                else:
                    logger.info(f"all file for {class_name} downloaded")
        else:
            logger.info("Using existing raw data")

    def split_data(self):
        # TODO: instead of splitting folders, need to split the paths
        if not os.path.exists(constants.SPLITDATA_DIR):
            logger.info(f"{constants.SPLITDATA_DIR} does not exist")

            logger.info("splitting the dataset")
            splitfolders.ratio(
                constants.RAWDATA_DIR,
                output=constants.SPLITDATA_DIR,
                seed=1337,
                ratio=(0.8, 0.1, 0.1),
            )
        else:
            logger.info(f"Using split data: {constants.SPLITDATA_DIR}")

        data_dict = {
            "train": os.path.join(constants.SPLITDATA_DIR, "train"),
            "val": os.path.join(constants.SPLITDATA_DIR, "val"),
            "test": os.path.join(constants.SPLITDATA_DIR, "test"),
        }
        return data_dict

    def get_customdataset(self):
        datadir_dict = self.split_data()
        image_datasets = {
            x: CustomDataset(datadir_dict[x], DATA_TRANSFORM[x]) for x in datadir_dict
        }

        return image_datasets
