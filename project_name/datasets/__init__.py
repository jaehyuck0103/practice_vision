from pydantic import BaseModel, Field

from .cifar import Cifar, CifarCfg
from .imagenet import Imagenet, ImagenetCfg


class DatasetCfg(BaseModel):
    specific: ImagenetCfg | CifarCfg = Field(..., discriminator="name")


def get_dataset(cfg: DatasetCfg, mode: str):
    if cfg.specific.name == "imagenet":
        dataset = Imagenet(cfg.specific, mode=mode)
        print(f"{len(dataset)} items\n")
    elif cfg.specific.name == "cifar":
        dataset = Cifar(cfg.specific, mode=mode)
    else:
        raise ValueError(f"Unexpected Dataset: name={cfg.specific.name}")
    return dataset
