"""
U-Net Biomedical Training Loop with Validation Checkpointing
Copyright (c) 2016-2018 Yousef Khaled.
"""

import torch
from ..models.unet_2d import UNet2D
from ..losses.dice_focal_loss import HybridDiceBCEloss


class UNetTrainer:
    def __init__(self, model: UNet2D, lr: float = 1e-4):
        self.model = model
        self.criterion = HybridDiceBCEloss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train_step(self, images: torch.Tensor, masks: torch.Tensor) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        logits = self.model(images)
        loss = self.criterion(logits, masks)
        loss.backward()
        self.optimizer.step()
        return loss.item()
