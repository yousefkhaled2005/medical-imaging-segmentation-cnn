"""
ONNX Model Export and TensorRT Inference Acceleration
Copyright (c) 2016-2018 Yousef Khaled.
"""

import torch
from ..models.unet_2d import UNet2D


def export_unet_onnx(output_path: str = "unet_brain_segmentation.onnx"):
    model = UNet2D(in_channels=1, out_channels=1, base_features=32)
    model.eval()
    dummy_input = torch.randn(1, 1, 256, 256)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input_mri_slice"],
        output_names=["predicted_mask_logits"],
        dynamic_axes={"input_mri_slice": {0: "batch_size"}, "predicted_mask_logits": {0: "batch_size"}}
    )
    print(f"Exported ONNX model successfully to {output_path}")


if __name__ == "__main__":
    export_unet_onnx()
