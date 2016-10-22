"""
Soft Dice Loss and Combined Hybrid Loss for Volumetric Lesion Segmentation
Copyright (c) 2016-2018 Yousef Khaled.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    """
    Differentiable surrogate of the Sørensen-Dice coefficient for severe class imbalance.
    """
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        cardinality = probs_flat.sum() + targets_flat.sum()
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice


class HybridDiceBCEloss(nn.Module):
    """
    Combines Binary Cross-Entropy with Soft Dice to balance pixel-level classification with structural overlap.
    """
    def __init__(self, weight_dice: float = 0.6, weight_bce: float = 0.4):
        super().__init__()
        self.weight_dice = weight_dice
        self.weight_bce = weight_bce
        self.dice = SoftDiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.weight_bce * bce_loss + self.weight_dice * dice_loss
