# ultralytics/models/sr/edsr.py

import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_feats, n_feats, 3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)

class EDSR(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feats=64, n_resblocks=4):
        super().__init__()
        self.head = nn.Conv2d(in_channels, n_feats, 3, padding=1)
        self.body = nn.Sequential(*[ResidualBlock(n_feats) for _ in range(n_resblocks)])
        self.tail = nn.Conv2d(n_feats, out_channels, 3, padding=1)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        x = self.tail(res)
        return x


import matplotlib.pyplot as plt
import os

def visualize_tensor_save(t, save_path="output.png"):
    """
    将 tensor 图像保存为 PNG 文件，用于无图形界面的服务器。

    Args:
        t (torch.Tensor): 形状为 [3, H, W] 或 [1, H, W] 的图像张量
        save_path (str): 保存路径，例如 "./debug_rgb.png"
    """
    img = t.detach().cpu().numpy()
    if img.shape[0] == 3:  # RGB
        img = img.transpose(1, 2, 0)
    elif img.shape[0] == 1:  # 单通道灰度
        img = img.squeeze(0)
    
    plt.imsave(save_path, img)
    print(f"[INFO] 图像已保存至: {os.path.abspath(save_path)}")

def debug_tensor_channels(t, name="Tensor"):
    print(f"[DEBUG] {name} shape: {t.shape}")
    for c in range(t.shape[1]):
        ch = t[:, c:c+1, :, :]
        minv = ch.min().item()
        maxv = ch.max().item()
        meanv = ch.mean().item()
        print(f"    Channel {c}: min={minv:.4f}, max={maxv:.4f}, mean={meanv:.4f}")

import torch


rgb = torch.rand(1,3,256,256)
ir  = torch.rand(1,3,256,256)
merged = torch.cat([rgb, ir], dim=1)  # [1,6,256,256]
print("merged min/max", merged.min().item(), merged.max().item())

edsr = EDSR()
out = edsr(rgb)   # should not be all zeros
print("edsr out min/max", out.min().item(), out.max().item())
state = torch.load("/public/home/cszx_mayan/lmz/lmz/DL/yolov11/YOLOFuse-main/checkpoints/edsr_epoch50.pth", map_location="cpu")
print({k: v.abs().sum().item() for k,v in state.items()})  # 检查是否全 0

