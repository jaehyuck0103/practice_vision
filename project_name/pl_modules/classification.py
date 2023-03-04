import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pydantic import BaseModel, StrictFloat, StrictInt
from torch import optim
from torchmetrics import MetricCollection
from torchmetrics.classification.accuracy import MulticlassAccuracy

from pl_template.utils.utils import get_epoch_lr
from project_name.modules import NetCfg, get_module


class OptimCfg(BaseModel):
    lr_list: list[StrictFloat]
    lr_milestones: list[StrictInt]


class PlClassificationCfg(BaseModel):
    num_classes: StrictInt
    optim: OptimCfg
    net: NetCfg


class PlClassification(pl.LightningModule):
    def __init__(self, cfg: PlClassificationCfg):
        super().__init__()
        self.cfg = cfg
        self.net = get_module(cfg.net)

        metrics = MetricCollection(
            {
                "err/top1": MulticlassAccuracy(num_classes=cfg.num_classes, average="micro"),
                "err/top5": MulticlassAccuracy(
                    num_classes=cfg.num_classes, average="micro", top_k=5
                ),
            }
        )
        self.train_metrics = metrics.clone()
        self.val_metrics = metrics.clone()

        self.loss_metrics = {}

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), 1.0)
        lr_scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda x: get_epoch_lr(x, self.cfg.optim.lr_list, self.cfg.optim.lr_milestones),
        )
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def training_step(self, inputs, batch_idx):

        # Forward pass
        y_pred = self.net(inputs)  # (batch, n_class)

        # Compute loss
        y_gt = inputs["label"]  # (batch)
        cur_loss = F.cross_entropy(y_pred, y_gt)

        losses = {
            "loss/test1": torch.tensor(1 / (batch_idx + 1), device=cur_loss.device),
            "loss/test2": torch.tensor(3.0, device=cur_loss.device),
            "total_loss": cur_loss,
        }

        return {"loss": cur_loss, "losses": losses, "preds": y_pred, "target": y_gt}

    def validation_step(self, inputs, batch_idx, dataloader_idx=0):

        # Forward pass
        y_gt = inputs["label"]  # (B)
        y_pred = self.net(inputs)  # (B, 10)

        return {"preds": y_pred, "target": y_gt}
