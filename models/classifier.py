from typing import Any
from config import Config
from utils import retrieve_class_from_string, visualize_attention
from torchvision.utils import make_grid
import lightning as L
import torch
import torch.nn as nn
import torchmetrics
import matplotlib.pyplot as plt
import tensorboard as tb

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

        num_classes = 17
        if cfg.model.head.params is not None:
            num_classes = cfg.model.head.params.get("out_features", 17)

        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.test_acc_5 = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes, top_k=5
        )

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
    
    def validation_step(self, batch, batch_idx):
        X, y = batch

        assert isinstance(X, torch.Tensor)

        if self.cfg.visualize_attention and batch_idx == 0:
            y_pred, attn_map = self.backbone(X, return_weights=True)
            y_pred = self.neck(y_pred)
            y_pred = self.head(y_pred)

            # Visualize attention
            # Normalize images first
            X = (X - X.min()) / (X.max() - X.min())
            masks = visualize_attention(attn_map, X.size(-1))  # [B, W, H]
            masks_grid = make_grid(masks.unsqueeze(1), nrow=X.size(0), normalize=False)
            imgs_grid = make_grid(X, nrow=X.size(0), normalize=False)
            masked_imgs_grid = make_grid(X * masks[:, None, :, :], nrow=X.size(0), normalize=False)
            full_grid = torch.cat([imgs_grid, masked_imgs_grid, masks_grid], dim=1)
            self.logger.experiment.add_image('attention_mask', full_grid, self.current_epoch)  # type: ignore

        else:
            y_pred = self.model(X)

        loss = self.criterion(y_pred, y)
        self.val_acc(y_pred, y)

        self.log_dict({"val_loss": loss, "val_acc": self.val_acc}, prog_bar=True)
        return loss
    
    def test_step(self, batch):
        X, y = batch
        y_pred = self.model(X)
        loss = self.criterion(y_pred, y)
        self.test_acc(y_pred, y)
        self.test_acc_5(y_pred, y)

        self.log_dict(
            {
                "test_loss": loss,
                "test_acc": self.test_acc,
                "test_acc_5": self.test_acc_5,
            },
            prog_bar=True,
        )
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
