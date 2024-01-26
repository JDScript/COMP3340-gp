from omegaconf import OmegaConf
import torch
import torch.nn as nn
import os
import torch.utils.tensorboard as tb
from time import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import instantiate_model, Model
from config import load_config, Config
from dataset import instentiate_dataloader
from utils import retrieve_class_from_string


def train(
    model: nn.Module,
    criterion: nn.Module,
    cfg: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    writer: tb.writer.SummaryWriter | None,
):
    optimizer: torch.optim.Optimizer = retrieve_class_from_string(
        cfg.trainer.optimizer.target
    )(model.parameters(), **cfg.trainer.optimizer.params)

    scheduler: torch.optim.lr_scheduler._LRScheduler = retrieve_class_from_string(
        cfg.trainer.scheduler.target
    )(optimizer, **cfg.trainer.scheduler.params)

    for epoch in range(1, cfg.trainer.epochs + 1):
        print(f"Epoch: {epoch}/{cfg.trainer.epochs}")

        epoch_losses = 0.0
        epoch_corrects = 0
        epoch_total = 0

        model.train()
        for X, y in tqdm(train_loader, "Training"):
            X = X.to(cfg.device)
            y = y.to(cfg.device)
            optimizer.zero_grad()
            y_pred = model(X)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            epoch_losses += loss.item() * y_pred.shape[0]
            epoch_corrects += torch.sum(torch.argmax(y_pred, dim=1) == y)
            epoch_total += X.shape[0]

        eval_losses = 0.0
        eval_corrects = 0
        eval_total = 0

        model.eval()
        with torch.no_grad():
            for X, y in tqdm(val_loader, "Evaluating"):
                X = X.to(cfg.device)
                y = y.to(cfg.device)
                y_pred = model(X)
                loss = criterion(y_pred, y)

                eval_losses += loss.item() * y_pred.shape[0]
                eval_corrects += torch.sum(torch.argmax(y_pred, dim=1) == y)
                eval_total += X.shape[0]

        if writer is not None:
            writer.add_scalar("Loss/train", epoch_losses / epoch_total, epoch)
            writer.add_scalar("Accuracy/train", epoch_corrects / epoch_total, epoch)
            writer.add_scalar("Loss/eval", eval_losses / eval_total, epoch)
            writer.add_scalar("Accuracy/eval", eval_corrects / eval_total, epoch)

        scheduler.step()
        print(
            f"loss: {epoch_losses/epoch_total}, acc: {epoch_corrects/epoch_total}, eval_loss: {eval_losses/eval_total}, eval_acc: {eval_corrects/eval_total}\n"
        )


def test(
    model: nn.Module,
    criterion: nn.Module,
    cfg: Config,
    test_loader: DataLoader,
    writer: tb.writer.SummaryWriter | None,
):
    model.eval()

    test_losses = 0.0
    test_corrects = 0
    test_total = 0
    with torch.no_grad():
        for X, y in tqdm(test_loader, "Testing"):
            X = X.to(cfg.device)
            y = y.to(cfg.device)
            y_pred = model(X)
            loss = criterion(y_pred, y)

            test_losses += loss.item() * y_pred.shape[0]
            test_corrects += torch.sum(torch.argmax(y_pred, dim=1) == y)
            test_total += X.shape[0]

    if writer is not None:
        writer.add_scalar("Loss/test", test_losses / test_total, 0)
        writer.add_scalar("Accuracy/test", test_corrects / test_total, 0)

    print(f"loss: {test_losses/test_total}, acc: {test_corrects/test_total}")


if __name__ == "__main__":
    conf = load_config()

    print(OmegaConf.to_yaml(conf))

    train_loader, val_loader, test_loader = instentiate_dataloader(conf)
    model = instantiate_model(conf).to(conf.device)
    criterion = nn.CrossEntropyLoss()

    if conf.device.startswith("cuda") and torch.cuda.device_count() > 1:
        print("Using {} GPUs".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    ckpt_loc = "./ckpt/run_{}".format(int(time()))
    os.mkdir(ckpt_loc)
    open(ckpt_loc + "/config.yaml", "w").write(OmegaConf.to_yaml(conf))

    writer = None
    if conf.tensorboard:
        writer = tb.writer.SummaryWriter(ckpt_loc)
        from tensorboard import program

        t = program.TensorBoard()
        t.configure(argv=[None, "--logdir", ckpt_loc, "--host", "0.0.0.0"])
        t.launch()

    train(
        model,
        criterion,
        conf,
        train_loader,
        val_loader,
        writer,
    )

    torch.save(model, ckpt_loc + "/checkpoint.ckpt")

    test(
        model,
        criterion,
        conf,
        test_loader,
        writer,
    )
