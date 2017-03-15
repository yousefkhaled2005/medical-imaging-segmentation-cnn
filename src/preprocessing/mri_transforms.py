"""
Volumetric MRI Slice Normalization and Elastic Data Augmentation
Copyright (c) 2016-2018 Yousef Khaled.
"""

import numpy as np


class MRIPreprocessor:
    """
    Clinical MRI intensity normalization (Z-score and min-max clipping) with spatial windowing.
    """
    @staticmethod
    def zscore_normalize(volume: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        if mask is not None and mask.sum() > 0:
            mean = volume[mask > 0].mean()
            std = volume[mask > 0].std()
        else:
            mean = volume.mean()
            std = volume.std()
        std = max(std, 1e-6)
        return (volume - mean) / std

    @staticmethod
    def clip_outliers(volume: np.ndarray, lower_pct: float = 1.0, upper_pct: float = 99.0) -> np.ndarray:
        p_low = np.percentile(volume, lower_pct)
        p_high = np.percentile(volume, upper_pct)
        return np.clip(volume, p_low, p_high)
