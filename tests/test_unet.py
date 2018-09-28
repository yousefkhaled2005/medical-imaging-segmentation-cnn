"""
Unit Tests for U-Net Architecture and Loss Functions
Copyright (c) 2016-2018 Yousef Khaled.
"""

import torch
import pytest
from src.models.unet_2d import UNet2D
from src.losses.dice_focal_loss import SoftDiceLoss, HybridDiceBCEloss


def test_unet_forward_shape():
    model = UNet2D(in_channels=1, out_channels=1, base_features=16)
    x = torch.randn(2, 1, 128, 128)
    out = model(x)
    assert out.shape == (2, 1, 128, 128)


def test_soft_dice_loss():
    loss_fn = SoftDiceLoss()
    logits = torch.ones(2, 1, 32, 32) * 5.0 # high probabilities
    targets = torch.ones(2, 1, 32, 32)
    loss = loss_fn(logits, targets)
    assert loss.item() < 0.1 # Near zero for perfect match
