from omegaconf import OmegaConf
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import instantiate_model
from config import load_config, Config
from dataset import instentiate_dataloader


def test(
    model: nn.Module,
    criterion: nn.Module,
    cfg: Config,
    test_loader: DataLoader,
):
    model.eval()

    test_losses = 0.0
    test_corrects = 0
    test_top5_corrects = 0
    test_total = 0
    with torch.no_grad():
        for X, y in tqdm(test_loader, "Testing"):
            X = X.to(cfg.device)
            y = y.to(cfg.device)
            y_pred = model(X)
            loss = criterion(y_pred, y)

            test_losses += loss.item() * y_pred.shape[0]
            test_corrects += torch.sum(torch.argmax(y_pred, dim=1) == y)
            _, top5_preds = torch.topk(y_pred, 5, dim=1)  # Get top-5 predictions
            test_top5_corrects += torch.sum(top5_preds == y.unsqueeze(1))
            test_total += X.shape[0]

    print(
        f"loss: {test_losses/test_total}, acc: {test_corrects/test_total}, top5_acc: {test_top5_corrects/test_total}"
    )


if __name__ == "__main__":
    conf = load_config()

    print(OmegaConf.to_yaml(conf))

    _, _, test_loader = instentiate_dataloader(conf)
    model = instantiate_model(conf).to(conf.device)
    criterion = nn.CrossEntropyLoss()

    test(
        model,
        criterion,
        conf,
        test_loader,
    )
