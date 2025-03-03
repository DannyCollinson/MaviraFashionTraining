""" Module defining the models used for Classification. """

from pathlib import Path
from typing import Any

from torch import Tensor, load
from torch import manual_seed as torch_set_seed
from torch.nn import Conv2d, Linear, Module, Sequential
from torch.nn.init import normal_, uniform_, zeros_
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_B3_Weights,
    VisionTransformer,
    efficientnet_b0,
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
    dropout = 0.1
    attention_dropout = 0.1
    representation_size = None

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
        if "dropout" in vit_kwargs:
            dropout = vit_kwargs["dropout"]
        if "attention_dropout" in vit_kwargs:
            attention_dropout = vit_kwargs["attention_dropout"]
        if "representation_size" in vit_kwargs:
            representation_size = vit_kwargs["representation_size"]
        if "num_classes" in vit_kwargs:
            num_classes = vit_kwargs["num_classes"]

    # create the model using PyTorch's VisionTransformer class
    vit = VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        mlp_dim=mlp_dim,
        dropout=dropout,
        attention_dropout=attention_dropout,
        representation_size=representation_size,
        num_classes=num_classes,
    )

    # initiialize the linear layer parameters
    init_range = 1.0 / vit.num_classes**0.5
    for submodule in vit.modules():
        if isinstance(submodule, Linear):
            normal_(submodule.weight, -init_range, init_range)
            zeros_(submodule.bias)
    uniform_(vit.heads.head.weight, -init_range, init_range)  # type: ignore

    return vit.to(device=get_device())


def create_efficientnet_b3(
    num_classes: int,
    weights: EfficientNet_B3_Weights | dict | str | Path | None = None,
    seed: int = DEFAULT_SEED,
) -> Module:
    """
    Create an EfficientNet-B3 model with the specified weights.

    Args:
        num_classes (int): number of classes in the dataset
        weights (EfficientNet_B3_Weights | dict | str | Path | None):
            weights for the model. If EfficientNet_B3_Weights, the model
            will attempt to load the weights. If dict, the dict is
            assumed to be a PyTorch state_dict without need to add a
            linear layer for the final output. If str, then the value
            "imagenet" will load the pre-trained IMAGENET1K_V1 weights
            from the torchvision.models.efficientnet module;
            otherwise, the str is assumed to be a path to the weights.
            If Path,then the path is assumed to be the path to the
            weights. If a string path or Path is provided, the model is
            assumed not to need a linear layer for the final output.
            If None, the model will be initialized with
            random weights. Defaults to None.
        seed (int): seed for initializing the weights.
            Defaults to DEFAULT_SEED (42).

    Returns:
        efficientnet_b3: the EfficientNet-B3 model that has been moved
            to the available device (cuda, mps, or cpu)
    """
    # create a linear layer for final output if using PyTorch default weights
    linear_out = Linear(in_features=1000, out_features=num_classes)
    init_range = 1.0 / linear_out.out_features**0.5
    uniform_(linear_out.weight, -init_range, init_range)
    zeros_(linear_out.bias)

    # create the model
    if isinstance(weights, EfficientNet_B3_Weights):
        # PyTorch pre-trained weights need a linear layer for the final output
        model = Sequential(efficientnet_b3(weights=weights), linear_out)
    elif weights == "imagenet":
        # PyTorch pre-trained weights need a linear layer for the final output
        model = Sequential(
            efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1),
            linear_out,
        )
    elif isinstance(weights, dict):
        # create a default model as a template to load the state_dict onto
        model = create_efficientnet_b3(num_classes=num_classes)  # type: ignore
        # load the state_dict onto the model
        model.load_state_dict(weights)
    elif isinstance(weights, (str, Path)):
        # create a default model as a template to load the state_dict onto
        model = create_efficientnet_b3(num_classes=num_classes)  # type: ignore
        # load the state_dict onto the model
        model.load_state_dict(load(weights, weights_only=True))
    else:
        # set seed for reproducibility
        torch_set_seed(seed)
        # create a model with random weights
        model = Sequential(efficientnet_b3(), linear_out)

    return model.to(device=get_device())


def create_efficientnet_b0_w_head(
    num_classes: int,
    weights: EfficientNet_B0_Weights | dict | str | Path | None = None,
    seed: int = DEFAULT_SEED,
    device: str = get_device(),
) -> Module:
    """
    Create an EfficientNet-B0 model with the specified weights
    and a linear layer for the final output.

    Args:
        num_classes (int): number of classes in the dataset
        weights (EfficientNet_B0_Weights | dict | str | Path | None):
            weights for the model. If EfficientNet_B0_Weights, the model
            will attempt to load the weights. If dict, the dict is
            assumed to be a PyTorch state_dict without need to add a
            linear layer for the final output. If str, then the value
            "imagenet" will load the pre-trained IMAGENET1K_V1 weights
            from the torchvision.models.efficientnet module;
            otherwise, the str is assumed to be a path to the weights.
            If Path,then the path is assumed to be the path to the
            weights. If a string path or Path is provided, the model is
            assumed not to need a linear layer for the final output.
            If None, the model will be initialized with
            random weights. Defaults to None.
        seed (int): seed for initializing the weights.
            Defaults to DEFAULT_SEED (42).
        device (str): device to move the model to.
            Defaults to the output of get_device().

    Returns:
        efficientnet_b0: the EfficientNet-B0 model that has been moved
            to the available device (cuda, mps, or cpu)
    """
    # create a linear layer for final output if using PyTorch default weights
    linear_out = Linear(in_features=1000, out_features=num_classes)
    init_range = 1.0 / linear_out.out_features**0.5
    uniform_(linear_out.weight, -init_range, init_range)
    zeros_(linear_out.bias)

    # create the model
    if isinstance(weights, EfficientNet_B0_Weights):
        # PyTorch pre-trained weights need a linear layer for the final output
        model = Sequential(efficientnet_b0(weights=weights), linear_out)
    elif weights == "imagenet":
        # PyTorch pre-trained weights need a linear layer for the final output
        model = Sequential(
            efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1),
            linear_out,
        )
    elif isinstance(weights, dict):
        # create a default model as a template to load the state_dict onto
        model = create_efficientnet_b0_w_head(
            num_classes=num_classes
        )  # type: ignore
        # load the state_dict onto the model
        model.load_state_dict(weights)
    elif isinstance(weights, (str, Path)):
        # create a default model as a template to load the state_dict onto
        model = create_efficientnet_b0_w_head(
            num_classes=num_classes
        )  # type: ignore
        # load the state_dict onto the model
        model.load_state_dict(load(weights, weights_only=True))
    else:
        # set seed for reproducibility
        torch_set_seed(seed)
        # create a model with random weights
        model = Sequential(efficientnet_b0(), linear_out)

    return model.to(device=device)


def create_efficientnet_b0(
    num_classes: int,
    weights: dict | str | Path | None = None,
    seed: int = DEFAULT_SEED,
    device: str = get_device(),
) -> Module:
    """
    Create an EfficientNet-B0 model with the specified weights.

    Args:
        num_classes (int): number of classes in the dataset
        weights (dict | str | Path | None): weights for the model.
            If dict, the dict is assumed to be a PyTorch state_dict.
            If str or Path, it is assumed to be a path to the weights.
            If None, the model will be initialized with random weights.
            Defaults to None.
        seed (int): seed for initializing the weights.
            Defaults to DEFAULT_SEED (42).
        device (str): device to move the model to.
            Defaults to the output of get_device().

    Returns:
        efficientnet_b0: the EfficientNet-B0 model that has been moved
            to the available device (cuda, mps, or cpu)
    """
    # create a linear layer for final output if using PyTorch default weights
    linear_out = Linear(in_features=1000, out_features=num_classes)
    init_range = 1.0 / linear_out.out_features**0.5
    uniform_(linear_out.weight, -init_range, init_range)
    zeros_(linear_out.bias)

    # create the model
    if isinstance(weights, EfficientNet_B0_Weights):
        # PyTorch pre-trained weights need a linear layer for the final output
        model = efficientnet_b0(weights=weights, num_classes=num_classes)
    elif weights == "imagenet":
        # PyTorch pre-trained weights need a linear layer for the final output
        model = efficientnet_b0(
            weights=EfficientNet_B0_Weights.IMAGENET1K_V1,
            num_classes=num_classes,
        )
    elif isinstance(weights, dict):
        # create a default model as a template to load the state_dict onto
        model = create_efficientnet_b0(num_classes=num_classes)  # type: ignore
        # load the state_dict onto the model
        model.load_state_dict(weights)
    elif isinstance(weights, (str, Path)):
        # create a default model as a template to load the state_dict onto
        model = create_efficientnet_b0(num_classes=num_classes)  # type: ignore
        # load the state_dict onto the model
        model.load_state_dict(load(weights, weights_only=True))
    else:
        # set seed for reproducibility
        torch_set_seed(seed)
        # create a model with random weights
        model = efficientnet_b0(num_classes=num_classes)

    return model.to(device=device)


class TestModel(Module):
    """
    Simple model for testing purposes
    """

    def __init__(self, input_dims: list[int], num_classes: int) -> None:
        super().__init__()

        # convolutional layer
        self.conv = Conv2d(3, 1, 16, 16)
        # initialize weights and biases
        normal_(self.conv.weight, mean=0.0, std=0.01)
        if self.conv.bias is not None:
            zeros_(self.conv.bias)

        # linear layer
        self.linear = Linear(
            int(
                self.conv.out_channels
                * input_dims[1]
                / self.conv.kernel_size[0]
                * input_dims[2]
                / self.conv.kernel_size[1]
            ),
            num_classes,
        )
        # initialize weights and biases
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
        fx = self.conv(x)

        fx = fx.flatten(start_dim=1)  # assuming batched input in dim 0
        fx = self.linear(fx)

        return fx  # CrossEntropyLoss includes softmax
