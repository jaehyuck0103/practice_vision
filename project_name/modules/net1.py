from typing import Literal

import torch
import torchvision
from pydantic import BaseModel
from torch import nn


class Net1Cfg(BaseModel):
    name: Literal["net1"]
    backbone: Literal["shufflenet"] | Literal["res34"] | Literal["res50"]


class Net1(nn.Module):
    def __init__(self, cfg: Net1Cfg):
        super().__init__()

        if cfg.backbone == "shufflenet":
            shuffle_net = torchvision.models.shufflenet_v2_x1_0(weights="DEFAULT")
            modules = list(shuffle_net.children())[:-1]
            modules.append(nn.AdaptiveAvgPool2d([1, 1]))
            self.conv_net = nn.Sequential(*modules)
            self.fc = nn.Linear(1024, 10)
        elif cfg.backbone == "res34":
            res34 = torchvision.models.resnet34(weights="DEFAULT")
            modules = list(res34.children())[:-1]
            self.conv_net = nn.Sequential(*modules)
            self.fc = nn.Linear(512, 10)
        elif cfg.backbone == "res50":
            res50 = torchvision.models.resnet50(weights="DEFAULT")
            modules = list(res50.children())[:-1]
            self.conv_net = nn.Sequential(*modules)
            self.fc = nn.Linear(2048, 10)
        else:
            raise ValueError(cfg.backbone)

    def snapshot_elements(self):
        return {
            "cnn_classifier": self,
        }

    def forward(self, inputs):
        x = inputs["img"]  # (B, C, H, W)

        # Normalize
        x = (x - 0.5) / 0.25

        # Forward Pass
        x = self.conv_net(x)
        x = torch.squeeze(x, 3)
        x = torch.squeeze(x, 2)
        x = self.fc(x)  # (B, 1)

        return x
