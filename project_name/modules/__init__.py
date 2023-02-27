from typing import Literal

from pydantic import BaseModel, Field

from .net1 import Net1, Net1Cfg
from .vit import ViT, ViTCfg


class Net2Cfg(BaseModel):
    name: Literal["net2"]


class NetCfg(BaseModel):
    specific: Net1Cfg | Net2Cfg | ViTCfg = Field(..., discriminator="name")


def get_module(cfg: NetCfg):
    if cfg.specific.name == "net1":
        dataset = Net1(cfg.specific)
    elif cfg.specific.name == "vit":
        dataset = ViT(cfg.specific)
    else:
        raise ValueError(f"Unexpected Module: name={cfg.specific.name}")
    return dataset
