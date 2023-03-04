from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from pydantic import BaseModel
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

DATASET_ROOT = {
    "train": Path("Data") / "ILSVRC2012" / "train",
    "val": Path("Data") / "ILSVRC2012" / "val",
}


class ImagenetCfg(BaseModel):
    name: Literal["imagenet"]


def _imread_float(f):
    img = cv2.imread(f).astype(np.float32) / 255
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.imread(f, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255

    return img


class Imagenet(Dataset):
    def __init__(self, cfg: ImagenetCfg, mode: str):
        super().__init__()

        # assert
        train_class_names = sorted(x.name for x in DATASET_ROOT["train"].iterdir() if x.is_dir())
        val_class_names = sorted(x.name for x in DATASET_ROOT["val"].iterdir() if x.is_dir())
        assert train_class_names == val_class_names
        self.class_name_to_id = {name: id for id, name in enumerate(train_class_names)}

        # Load
        self.mode = mode
        root = DATASET_ROOT[mode]
        self.img_paths = list(root.glob("*/*.JPEG"))

    def __getitem__(self, _idx):

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
        if self.mode == "train":
            return len(self.img_paths) * 10
        else:
            return len(self.img_paths)
