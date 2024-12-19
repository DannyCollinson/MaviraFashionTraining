""" Custom PyTorch transform functions. """

from torch import Tensor


def rescale01(tensor: Tensor) -> Tensor:
    """
    Rescale the input tensor to have values between 0 and 1.

    Args:
        tensor (Tensor): input tensor to rescale

    Returns:
        Tensor: rescaled tensor
    """
    return (tensor - tensor.min()) / (tensor.max() - tensor.min())
