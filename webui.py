import gradio as gr
import torch
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
import torch.utils
import torch.utils.data
from config import Config
from models.classifier import Classifier
from fvcore.nn import FlopCountAnalysis
from datasets import instentiate_dataloader, Flowers17
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF
import pytorch_lightning.accelerators.accelerator as accelerator
import random


log_path = Path(__file__).parent / "lightning_logs"
checkpoints = []
cfg_schema = OmegaConf.structured(Config)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


def refresh_checkpoints():
    global checkpoints
    checkpoints = [str(log) for log in log_path.glob("version_*/checkpoints/*.ckpt")]


def load_model(ckpt_path: str):
    model = Classifier.load_from_checkpoint(ckpt_path, map_location="cpu")
    train_loader, val_loader, test_loader = instentiate_dataloader(model.cfg)
    gr.Info("Checkpoint loaded: {}".format(model.cfg.model.backbone.target))
    return model, train_loader, val_loader, test_loader, OmegaConf.to_yaml(model.cfg)


def visualize_images(images: torch.Tensor):
    return TF.to_pil_image(make_grid(images, nrow=4, normalize=True))


def sample_from_dataloader(test_loader: torch.utils.data.DataLoader):
    rand_idx = random.randint(0, len(test_loader.dataset) - 1)  # type: ignore
    data = test_loader.dataset[rand_idx]
    return (
        data,
        TF.to_pil_image(make_grid(data[0], nrow=1, normalize=True)),
        Flowers17._classes[data[1]],
    )


def make_inference(model: Classifier, data: tuple[torch.Tensor, np.ndarray]):
    gr.Info("Inference using {}".format(device))
    model.eval()
    model.to(device)
    X, y = data
    X = X.unsqueeze(0).to(device)
    pred: torch.Tensor = model(X)

    # Log Softmax
    if pred.min() < 0:
        pred = torch.exp(pred)

    pred_np = pred.squeeze().detach().cpu().numpy()  # [17]
    pred_dict = {Flowers17._classes[i]: pred_np[i] for i in range(len(pred_np))}
    return pred_dict


def show_from_dataloader(
    loader_type: str,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
):
    assert loader_type in ["Train", "Validation", "Test"]
    assert (
        train_loader is not None and val_loader is not None and test_loader is not None
    ), "Load a model first."
    loader = (
        train_loader
        if loader_type == "Train"
        else val_loader if loader_type == "Validation" else test_loader
    )
    total = len(loader.dataset)  # type: ignore
    random_indices = torch.randint(
        low=0,
        high=total,
        size=(8,),
    ).tolist()

    data = [loader.dataset[i] for i in random_indices]
    images_grid = TF.to_pil_image(
        make_grid([d[0] for d in data], nrow=4, normalize=True)
    )

    return images_grid


refresh_checkpoints()

with gr.Blocks("COMP3340 Group Project Demo") as demo:
    gr.Markdown("# COMP3340 Group Project Demo - Image Classifier on Flowers12")
    model = gr.State()
    config = gr.State()
    data = gr.State()
    train_loader = gr.State()
    val_loader = gr.State()
    test_loader = gr.State()

    with gr.Tab("Inference"):
        with gr.Row():
            with gr.Column():
                sample = gr.Image(interactive=False, label="Sample Image")
                sample_class = gr.Label(label="Sample Ground Truth Class")
                draw_sample_btn = gr.Button("Draw Sample", size="sm")
                draw_sample_btn.click(
                    sample_from_dataloader,
                    inputs=[test_loader],
                    outputs=[data, sample, sample_class],
                )
            with gr.Column():
                pred = gr.Label(
                    label="Predictions",
                )
        inference_btn = gr.Button("Inference", size="sm")
        inference_btn.click(make_inference, inputs=[model, data], outputs=[pred])
    with gr.Tab("Dataset"):
        loader = gr.Dropdown(
            label="Dataset",
            choices=[
                "Train",
                "Validation",
                "Test",
            ],
        )
        sample_images = gr.Image(interactive=False)
        sample_btn = gr.Button("Sample", size="sm")
        sample_btn.click(
            show_from_dataloader,
            inputs=[loader, train_loader, val_loader, test_loader],
            outputs=[sample_images],
        )

    with gr.Tab("Settings"):
        checkpoint = gr.Dropdown(
            label="Model Checkpoint",
            choices=checkpoints,
        )
        with gr.Row():
            load_btn = gr.Button("Load", size="sm")

            refresh_btn = gr.Button(
                "Refresh",
                size="sm",
                every=5,
            )
            refresh_btn.click(refresh_checkpoints)
        model_config_viewer = gr.Code(
            label="Model Configuration",
            language="yaml",
        )
        load_btn.click(
            load_model,
            inputs=[checkpoint],
            outputs=[model, train_loader, val_loader, test_loader, model_config_viewer],
        )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
