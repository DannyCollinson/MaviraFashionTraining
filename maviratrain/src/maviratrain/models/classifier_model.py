""" Module defining the models used for Classification. """

from typing import Any

from torch import Tensor, manual_seed as torch_set_seed
from torch.nn import BatchNorm1d, Linear, Module, ReLU, Sequential
from torch.nn.init import normal_, uniform_, zeros_
from torchvision.models import (
    EfficientNet_B3_Weights,
    VisionTransformer,
    efficientnet_b3,
)

from maviratrain.utils.constants import DEFAULT_SEED
from maviratrain.utils.general import get_device


def create_vit(vit_kwargs: dict[str, Any] | None = None) -> VisionTransformer:
    """
    Create a ViT model with the given kwargs as defined here:
    https://github.com/pytorch/vision/blob/a59c93980d97f6216917415ae25f3ac88e64cbb4/torchvision/models/vision_transformer.py#L160

    Args:
        vit_kwargs (dict[str, Any]): kwargs for the PyTorch ViT class

    Returns:
        VisionTransformer: the specified ViT model that has been moved
            to the available device (cuda, mps, or cpu)
    """
    # set default values
    image_size = 224
    patch_size = 16
    num_layers = 24
    num_heads = 16
    hidden_dim = 1024
    mlp_dim = 4096
    num_classes = 4

    # update default values with kwargs
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


def create_efficientnet_b3(
    num_classes: int,
    weights: EfficientNet_B3_Weights | None = None,
    seed: int = DEFAULT_SEED,
) -> Module:
    """
    Create an EfficientNet-B3 model with the given weights.

    Returns:
        num_classes (int): number of classes in the dataset
        efficientnet_b3: the EfficientNet-B3 model that has been moved
            to the available device (cuda, mps, or cpu)
        seed (int): seed for initializing the weights
    """
    torch_set_seed(seed)

    linear_out = Linear(in_features=1000, out_features=num_classes)
    init_range = 1.0 / linear_out.out_features**0.5
    uniform_(linear_out.weight, -init_range, init_range)
    zeros_(linear_out.bias)

    if weights is None:
        model = Sequential(
            efficientnet_b3(),
            linear_out,
        )
    else:
        model = Sequential(
            efficientnet_b3(weights=weights),
            linear_out,
        )

    return model.to(device=get_device())


class SimpleModel(Module):
    """
    Simple model for testing purposes
    """

    def __init__(self, input_dims: list[int], num_classes: int) -> None:
        super().__init__()

        # calculate the number of input features
        input_features = 1
        for dim in input_dims:
            input_features *= dim

        self.linear1 = Linear(in_features=input_features, out_features=1024)
        self.linear2 = Linear(in_features=1024, out_features=1024)
        self.linear3 = Linear(in_features=1024, out_features=1024)
        self.linear4 = Linear(in_features=1024, out_features=1024)
        self.linear5 = Linear(in_features=1024, out_features=num_classes)

        self.relu = ReLU()

        self.batch_norm = BatchNorm1d(num_features=1024)

        normal_(self.linear1.weight, mean=0.0, std=0.01)
        zeros_(self.linear1.bias)
        normal_(self.linear2.weight, mean=0.0, std=0.01)
        zeros_(self.linear2.bias)
        normal_(self.linear3.weight, mean=0.0, std=0.01)
        zeros_(self.linear3.bias)
        normal_(self.linear4.weight, mean=0.0, std=0.01)
        zeros_(self.linear4.bias)
        normal_(self.linear5.weight, mean=0.0, std=0.01)
        zeros_(self.linear5.bias)

        self.to(device=get_device())

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the model

        Args:
            x (Tensor): input tensor

        Returns:
            Tensor: output tensor
        """
        x = x.flatten(start_dim=1)  # assuming batched input in dim 0

        fx = self.batch_norm(x)
        fx = self.linear1(fx)
        fx = self.relu(fx)

        fx = self.batch_norm(fx)
        fx = self.linear2(fx)
        fx = self.relu(fx)

        fx = self.batch_norm(fx)
        fx = self.linear3(fx)
        fx = self.relu(fx)

        fx = self.batch_norm(fx)
        fx = self.linear4(fx)
        fx = self.relu(fx)

        fx = self.linear5(fx)

        return fx  # CrossEntropyLoss includes softmax


class TestModel(Module):
    """
    Simple model for testing purposes
    """

    def __init__(self, input_dims: list[int], num_classes: int) -> None:
        super().__init__()

        # calculate the number of input features
        num_input_features = 1
        for dim in input_dims:
            num_input_features *= dim

        self.linear = Linear(
            in_features=num_input_features, out_features=num_classes
        )
        normal_(self.linear.weight, mean=0.0, std=0.01)
        zeros_(self.linear.bias)

        self.to(device=get_device())

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the model

        Args:
            x (Tensor): input tensor

        Returns:
            Tensor: output tensor
        """
        x = x.flatten(start_dim=1)  # assuming batched input in dim 0

        fx = self.linear(x)

        return fx  # CrossEntropyLoss includes softmax
