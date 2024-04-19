import torch
import torch.nn.functional as F
import numpy as np


def visualize_attention(
    attn_map: list[torch.Tensor], img_size: int = 224
) -> torch.Tensor:
    # attn_map [batch_size, num_heads, W, H] * num_layers
    attn = torch.stack(attn_map)  # [num_layers, batch_size, num_heads, W, H]
    # take average over all heads
    attn = attn.mean(dim=2)  # [num_layers, batch_size, W, H]
    # residual
    # attn = attn 
    # normalize for each layer
    attn = attn / attn.sum(dim=(2, 3), keepdim=True)  # [num_layers, batch_size, W, H]

    joint_attentions = torch.zeros_like(attn)  # [num_layers, batch_size, W, H]
    for i in range(1, attn.size(0)):
        joint_attentions[i] = torch.bmm(attn[i], attn[i - 1])
    v = joint_attentions[-1]  # [batch_size, W, H]

    grid_size = int(np.sqrt(v.size(-1)))
    masks = v[:, 0, 1:].reshape(
        -1, grid_size, grid_size
    )  # [batch_size, W_patch_num, H_patch_num]
    # Interpolate to img_size
    masks: torch.Tensor = F.interpolate(
        masks.unsqueeze(1),
        size=(img_size, img_size),
        mode="bilinear",
        align_corners=False,
    )  # [B, 1, W, H]
    masks = masks.squeeze(1)  # [B, W, H]
    # Normalize
    masks /= masks.amax(dim=(1, 2), keepdim=True)  # [B, W, H]

    return masks
