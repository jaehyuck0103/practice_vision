from typing import Literal

from pydantic import BaseModel, Field

from .cifar import Cifar10Cfg, get_cifar
from .dataset1 import Dataset1, Dataset1Cfg


class Dataset2Cfg(BaseModel):
    name: Literal["dataset2"]


class DatasetCfg(BaseModel):
    specific: Dataset1Cfg | Dataset2Cfg | Cifar10Cfg = Field(..., discriminator="name")


def get_dataset(cfg: DatasetCfg, mode: str):
    if cfg.specific.name == "dataset1":
        dataset = Dataset1(cfg.specific, mode=mode)
        print(f"{len(dataset)} items\n")
    elif cfg.specific.name == "cifar10":
        dataset = get_cifar(mode=mode)
    else:
        raise ValueError(f"Unexpected Dataset: name={cfg.specific.name}")
    return dataset
