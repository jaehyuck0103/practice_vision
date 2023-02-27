from pathlib import Path

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only
from torch.utils.tensorboard.writer import SummaryWriter


class ScalarTensorboardCallback(Callback):
    def __init__(self):
        pass

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        self.train_writer = SummaryWriter(Path(trainer.default_root_dir) / "train0")
        self.val_writers = [
            SummaryWriter(Path(trainer.default_root_dir) / f"val{x}")
            for x in range(len(trainer.val_dataloaders))
        ]

    @rank_zero_only
    def on_train_epoch_start(self, trainer, pl_module):
        self.train_writer.add_scalar(
            "lr", pl_module.lr_schedulers().get_last_lr()[0], trainer.current_epoch
        )

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx + 1 == trainer.num_training_batches:
            for key, val in pl_module.loss_metrics.items():
                self.train_writer.add_scalar(key, val.compute().item(), trainer.current_epoch)
            for key, val in pl_module.train_metrics.compute().items():
                self.train_writer.add_scalar(key, val.item(), trainer.current_epoch)

    @rank_zero_only
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if batch_idx + 1 == trainer.num_val_batches[dataloader_idx]:
            for key, val in pl_module.val_metrics.compute().items():
                self.val_writers[dataloader_idx].add_scalar(key, val.item(), trainer.current_epoch)
