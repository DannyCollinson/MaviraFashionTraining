"""Defines the model used for Classifier training"""

from typing import Any, Union

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
from torchvision.models import VisionTransformer

from maviratrain.utils.general import get_device


def create_vit(
    vit_kwargs: Union[dict[str, Any], None] = None
) -> VisionTransformer:
    """
    Create a ViT model with the given kwargs as defined here:
    https://github.com/pytorch/vision/blob/a59c93980d97f6216917415ae25f3ac88e64cbb4/torchvision/models/vision_transformer.py#L160

    Args:
        vit_kwargs (dict[str, Any]): kwargs for the PyTorch ViT class

    Returns:
        VisionTransformer: the specified ViT model that has been moved to the
            available device (cuda or cpu)
    """
    image_size = 224
    patch_size = 16
    num_layers = 24
    num_heads = 16
    hidden_dim = 1024
    mlp_dim = 4096
    num_classes = 4

    if vit_kwargs is not None:
        if "image_size" in vit_kwargs:
            image_size = vit_kwargs["image_size"]
        if "patch_size" in vit_kwargs:
            patch_size = vit_kwargs["patch_size"]
        if "num_layers" in vit_kwargs:
            num_layers = vit_kwargs["num_layers"]
        if "num_heads" in vit_kwargs:
            num_heads = vit_kwargs["num_heads"]
        if "hidden_dim" in vit_kwargs:
            hidden_dim = vit_kwargs["hidden_dim"]
        if "mlp_dim" in vit_kwargs:
            mlp_dim = vit_kwargs["mlp_dim"]
        if "num_classes" in vit_kwargs:
            num_classes = vit_kwargs["num_classes"]

    vit = VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        mlp_dim=mlp_dim,
        num_classes=num_classes,
    )

    return vit.to(device=get_device())
