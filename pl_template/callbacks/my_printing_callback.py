import datetime
import time

import torch
from pytorch_lightning.callbacks import Callback
from torchmetrics import MeanMetric

from pl_template.utils.utils import AverageMeter, StaticPrinter, setup_for_distributed


class MyPrintingCallback(Callback):
    def __init__(self):
        self.sp = StaticPrinter()
        self.epoch_start = time.time()
        self.batch_end = time.time()

        self.iter_time = AverageMeter()
        self.data_time = AverageMeter()

    @staticmethod
    def to_str(*args):
        metric_dict = {}
        for arg in args:
            if isinstance(arg, dict):
                for key, val in arg.items():
                    metric_dict[key] = val.compute().item()
            else:
                for key, val in arg.compute().items():
                    metric_dict[key] = val.item()

        return ", ".join(f"{key}: {val:.4f}" for key, val in metric_dict.items())

    def setup(self, trainer, pl_module, stage: str):
        setup_for_distributed(trainer.is_global_zero)

    def on_train_epoch_start(self, trainer, pl_module):
        self.sp = StaticPrinter()
        self.epoch_start = time.time()
        pl_module.train_metrics.reset()

        print("\n\n-----------------------------")
        print(
            f"Epoch {pl_module.current_epoch} - LR {pl_module.lr_schedulers().get_last_lr()[0]:.1e}"
        )
        print("")

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.data_time.update(time.time() - self.batch_end)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):

        self.iter_time.update(time.time() - self.batch_end)
        self.batch_end = time.time()
        losses = outputs["losses"]

        if batch_idx == 0:
            pl_module.loss_metrics = {
                key: MeanMetric().to(losses["total_loss"].device) for key in losses.keys()
            }

        for key, val in losses.items():
            pl_module.loss_metrics[key].update(val)

        curr_err = pl_module.train_metrics(outputs["preds"], outputs["target"])

        if (batch_idx + 1) % 10 == 0:
            self.sp.reset()
            self.sp.print(
                f"Epoch {trainer.current_epoch} | Training [{batch_idx+1}/{trainer.num_training_batches}]"
            )
            self.sp.print(
                f"iter_time: {self.iter_time.val:.4f}, data_time: {self.data_time.val:.4f}"
            )
            self.iter_time.reset()
            self.data_time.reset()

            self.sp.print(f"max mem (MB): {torch.cuda.max_memory_allocated() / 2**20:.0f}")
            for key, val in pl_module.loss_metrics.items():
                self.sp.print(f"{key}: {losses[key]:.6f} ({val.compute():.6f})")
            for key, val in pl_module.train_metrics.compute().items():
                self.sp.print(f"{key}: {curr_err[key]:.6f} ({val.item():.6f})")

        if batch_idx + 1 == trainer.num_training_batches:
            print(
                f"Epoch {trainer.current_epoch} | Training"
                f" | epoch summary: {self.to_str(pl_module.loss_metrics, pl_module.train_metrics)}"
            )
            epoch_duration = time.time() - self.epoch_start
            print("Train Epoch Duration: ", str(datetime.timedelta(seconds=epoch_duration)))
            # pl_module.log_dict(pl_module.train_metrics.compute())

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):

        metrics = pl_module.val_metrics
        num_batches = trainer.num_val_batches[dataloader_idx]

        if batch_idx == 0:
            self.sp = StaticPrinter()
            metrics.reset()
            print("")

        curr_err = metrics(outputs["preds"], outputs["target"])

        if (batch_idx + 1) % 10 == 0:
            self.sp.reset()
            self.sp.print(
                f"Epoch {trainer.current_epoch} | Validation [{batch_idx+1}/{num_batches}]"
            )
            self.sp.print(f"max mem (MB): {torch.cuda.max_memory_allocated() / 2**20:.0f}")
            for key, val in metrics.compute().items():
                self.sp.print(f"{key}: {curr_err[key]:.6f} ({val.item():.6f})")

        if batch_idx + 1 == num_batches:
            print(
                f"Epoch {trainer.current_epoch} | Validation"
                f" | epoch summary: {self.to_str(metrics)}"
            )
            # pl_module.log_dict(metrics.compute())
