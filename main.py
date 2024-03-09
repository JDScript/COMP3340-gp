import argparse
import torch
import lightning as L
from config import load_config
from models.classifier import Classifier
from datasets import instentiate_dataloader
from fvcore.nn import FlopCountAnalysis, flop_count_table

parser = argparse.ArgumentParser(
    prog="python main.py",
)
parser.add_argument(
    "-config", help="additional config file path", default="./configs/_base.yaml"
)
args = parser.parse_args()
cfg = load_config(args.config)


if __name__ == "__main__":
    model = Classifier(cfg=cfg)
    flops = FlopCountAnalysis(model, inputs=torch.zeros(1, 3, 224, 224))
    print(flop_count_table(flops))
    train_loader, val_loader, test_loader = instentiate_dataloader(cfg)
    trainer = L.Trainer(max_epochs=cfg.trainer.epochs)
    trainer.fit(model, train_loader, val_loader)
