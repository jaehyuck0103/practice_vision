from pathlib import Path
from typing import Literal

from pydantic import BaseModel
from torchvision import transforms
from torchvision.datasets import CIFAR10


class Cifar10Cfg(BaseModel):
    name: Literal["cifar10"]


test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            [0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784]
        ),
    ]
)
# For training, we add some augmentation. Networks are too powerful and would overfit.
train_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784]
        ),
    ]
)


def get_cifar(mode: str):

    DATASET_PATH = Path("Data")

    if mode == "train":
        return CIFAR10(root=DATASET_PATH, train=True, transform=train_transform, download=True)
    elif mode == "val":
        return CIFAR10(root=DATASET_PATH, train=False, transform=test_transform, download=True)
    else:
        raise ValueError(mode)
