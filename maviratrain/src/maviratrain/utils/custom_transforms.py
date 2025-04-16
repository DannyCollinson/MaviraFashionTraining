"""Custom PyTorch transform functions."""

from torch import Tensor, float32
from torch.nn import Sequential
from torchvision.transforms.v2 import Normalize, ToDtype


def rescale01(tensor: Tensor) -> Tensor:
    """
    Rescale the input tensor to have values between 0 and 1.

    Args:
        tensor (Tensor): input tensor to rescale

    Returns:
        Tensor: rescaled tensor
    """
    return (tensor - tensor.min()) / (tensor.max() - tensor.min())


class ImageNetNormalize(Sequential):
    """
    A set of transforms that performs standard ImageNet normalization

    Input is converted to float dtype with values in the range [0, 1]
    and then normalized using the ImageNet-1k statistics
    `mean=[0.485, 0.456, 0.406]` and `std=[0.229, 0.224, 0.225]`.
    """

    def __init__(self):
        super().__init__()

        # define transforms
        self.todtype = ToDtype(float32, scale=True)
        self.normalize = Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.transform = Sequential(self.todtype, self.normalize)

    def imagenet_normalize(self, tensor: Tensor) -> Tensor:
        """
        Converts input to float dtype with values in the range [0, 1]
        and then normalizes inputs using the ImageNet-1k statistics
        `mean=[0.485, 0.456, 0.406]` and `std=[0.229, 0.224, 0.225]`

        Args:
            tensor (Tensor): tensor to be normalized

        Returns:
            Tensor: tensor normalized using ImageNet-1k statistics
        """
        return self.transform(tensor)
