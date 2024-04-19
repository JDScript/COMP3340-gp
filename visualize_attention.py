import argparse
import torch
import lightning as L
from config import load_config
from models.classifier import Classifier
from datasets import instentiate_dataloader
from fvcore.nn import FlopCountAnalysis, flop_count_table
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

from torchvision.transforms import transforms

parser = argparse.ArgumentParser(
    prog="python main.py",
)
parser.add_argument(
    "-config", help="additional config file path", default="./configs/_base.yaml"
)
parser.add_argument(
    "-checkpoint",
    help="checkpoint file path",
    default=None,
)
args = parser.parse_args()
cfg = load_config(args.config)

if __name__ == "__main__":
    model = Classifier(cfg)
    if args.checkpoint:
        model = Classifier.load_from_checkpoint(args.checkpoint, map_location="cpu")
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    img = Image.open("./data/flowers-17/jpg/image_0001.jpg")

    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    x = transform(img)

    out, att_mat = model.backbone(x.unsqueeze(0))


    att_mat = torch.stack(att_mat)

    att_mat = torch.mean(att_mat, dim=0)

    print(att_mat.size())

    residual_att = torch.eye(att_mat.size(1))

    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])

    # Attention from the output token to the input space.
    v = joint_attentions[-1]
    grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
    mask = cv2.resize(mask / mask.max(), img.size)[..., np.newaxis]
    result = (mask * img).astype("uint8")

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))

    ax1.set_title("Original")
    ax2.set_title("Attention Map")
    _ = ax1.imshow(img)
    _ = ax2.imshow(result)

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))
    for i, v in enumerate(joint_attentions):
      if i <= 10:
        continue
      # Attention from the output token to the input space.
      mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
      mask = cv2.resize(mask / mask.max(), img.size)[..., np.newaxis]
      result = (mask * img).astype("uint8")

      # fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))
      ax1.set_title('Original')
      ax2.set_title('Attention Map_%d Layer' % (i+1))
      _ = ax1.imshow(img)
      _ = ax2.imshow(result)
