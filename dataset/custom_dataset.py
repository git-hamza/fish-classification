
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import datasets, models, transforms
from torchvision.transforms import v2
import random

# with Augmentation
class CustomDataset(datasets.ImageFolder):
    def __init__(self, data_dir, transform=None, augmentation=True):
        super().__init__(
            data_dir,
            transform=transform
        )

        self.augmentation_transform = [
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.RandomRotation(10),
            v2.ColorJitter(),
            v2.GaussianBlur(kernel_size=(5, 9)),
            v2.GaussianNoise()
        ]
        
        if augmentation:
            self.samples_with_aug = [(el, "normal") for el in self.samples]
            self.samples_with_aug.extend([(el, "aug") for el in self.samples])
        else:
            self.samples_with_aug = self.samples

    def __len__(self) -> int:
        return len(self.samples_with_aug)

    def __getitem__(self, index):
        item, flag = self.samples_with_aug[index]
        path, target = item

        sample = self.loader(path)

        if self.transform is not None:
            if flag == "aug":
                random_aug = random.randint(0, len(self.augmentation_transform)-1)
                transform = v2.Compose(self.transform.transforms + [self.augmentation_transform[random_aug]])
            else:
                transform = self.transform

            sample = transform(sample)

        return sample, target
    

if __name__ == "__main__":
    from dataset.loading_data import DATA_TRANSFORM
    custom_data = CustomDataset(data_dir="/Users/hash/hamza_data/Work/github/fish-classification/data/split_data/test", transform=DATA_TRANSFORM["test"])
    dataloader = torch.utils.data.DataLoader(custom_data)
    for inputs, classes in dataloader:
        print(classes)
    print("data")