import math
import random
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from pydantic import BaseModel, StrictFloat
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

DATASET_ROOT = {
    "train": Path("Data") / "ILSVRC2012" / "train",
    "val": Path("Data") / "ILSVRC2012" / "val",
}


class ImagenetCfg(BaseModel):
    name: Literal["imagenet"]
    mode: Literal["train"] | Literal["val"]
    epoch_scale_factor: StrictFloat


def _imread_float(f):
    img = cv2.imread(f).astype(np.float32) / 255
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.imread(f, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255

    return img


class Imagenet(Dataset):
    def __init__(self, cfg: ImagenetCfg):
        super().__init__()

        # assert
        train_class_names = sorted(x.name for x in DATASET_ROOT["train"].iterdir() if x.is_dir())
        val_class_names = sorted(x.name for x in DATASET_ROOT["val"].iterdir() if x.is_dir())
        assert train_class_names == val_class_names
        self.class_name_to_id = {name: id for id, name in enumerate(train_class_names)}

        # Load
        self.mode = cfg.mode
        root = DATASET_ROOT[cfg.mode]
        self.img_paths = list(root.glob("*/*.JPEG"))

        #
        self.epoch_scale_factor = cfg.epoch_scale_factor

    def __getitem__(self, _idx):

        if self.epoch_scale_factor < 1:
            _idx += len(self) * random.randrange(math.ceil(1 / self.epoch_scale_factor))

        img_path = self.img_paths[_idx % len(self.img_paths)]
        label = self.class_name_to_id[img_path.parent.name]

        img = _imread_float(str(img_path))
        img = cv2.resize(img, (224, 224))

        if self.mode == "train":
            # some augmentation
            pass
        else:
            # no augmentation
            pass

        # Add depth channels
        img = to_tensor(img)

        # sample return
        sample = {"img": img, "label": label}

        return sample

    def __len__(self):
        return round(len(self.img_paths) * self.epoch_scale_factor)
