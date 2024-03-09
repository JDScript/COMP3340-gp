from typing import Any
from config import Config
from utils.cls import retrieve_class_from_string
import lightning as L
import torch
import torch.nn as nn
import torchmetrics


class Classifier(L.LightningModule):
    def __init__(
        self,
        cfg: Config,
    ):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(cfg)
        self.backbone: nn.Module = retrieve_class_from_string(
            cfg.model.backbone.target
        )(**cfg.model.backbone.params)
        self.neck: nn.Module = retrieve_class_from_string(cfg.model.neck.target)(
            **cfg.model.neck.params
        )
        self.head: nn.Module = retrieve_class_from_string(cfg.model.head.target)(
            **cfg.model.head.params
        )
        self.model = nn.Sequential(
            self.backbone,
            self.neck,
            self.head,
        )
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=17)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=17)

    def forward(self, x: torch.Tensor) -> Any:
        return self.model(x)

    def training_step(self, batch):
        X, y = batch
        y_pred = self.model(X)
        loss = self.criterion(y_pred, y)
        self.train_acc(y_pred, y)

        self.log_dict(
            {"train_loss": loss, "train_acc": self.train_acc},
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch):
        X, y = batch
        y_pred = self.model(X)
        loss = self.criterion(y_pred, y)
        self.val_acc(y_pred, y)

        self.log_dict({"val_loss": loss, "val_acc": self.val_acc}, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer: torch.optim.Optimizer = retrieve_class_from_string(
            self.cfg.trainer.optimizer.target
        )(self.parameters(), **self.cfg.trainer.optimizer.params)

        scheduler: torch.optim.lr_scheduler._LRScheduler = retrieve_class_from_string(
            self.cfg.trainer.scheduler.target
        )(optimizer, **self.cfg.trainer.scheduler.params)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }
