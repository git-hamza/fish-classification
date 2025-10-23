import random

from torchvision import datasets
from torchvision.transforms import v2


# with Augmentation
class CustomDataset(datasets.ImageFolder):
    def __init__(
        self,
        data_dir: str,
        transform: v2.Compose | None = None,
        augmentation: bool = False,
    ) -> None:
        super().__init__(data_dir, transform=transform)
        self.augmentation = augmentation
        self.augmentation_transform = [
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.RandomRotation(10),
            v2.ColorJitter(),
            v2.GaussianBlur(kernel_size=(5, 9)),
            v2.GaussianNoise(),
        ]
        if self.augmentation:
            self.samples_with_aug = [(el, "normal") for el in self.samples]
            self.samples_with_aug.extend([(el, "aug") for el in self.samples])
        else:
            self.samples_with_aug = self.samples

    def __len__(self) -> int:
        return len(self.samples_with_aug)

    def __getitem__(self, index: int):
        if self.augmentation:
            path, target, flag = self.samples_with_aug[index]
        else:
            flag = "normal"
            path, target = self.samples_with_aug[index]

        sample = self.loader(path)

        if self.transform is not None:
            if flag == "aug":
                random_aug = random.randint(0, len(self.augmentation_transform) - 1)
                transform = v2.Compose(
                    self.transform.transforms
                    + [self.augmentation_transform[random_aug]]
                )
            else:
                transform = self.transform

            sample = transform(sample)

        return sample, target
