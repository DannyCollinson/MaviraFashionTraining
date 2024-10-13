""" Contains constants used in the project. """

from numpy import load as np_load
from torch import load as torch_load
from torchvision.io import decode_image

# define loading functions for different image formats
LOAD_FUNCS = {
    "torchvision.io.decode_image": decode_image,
    "torch.load": torch_load,
    "numpy.load": np_load,
}
