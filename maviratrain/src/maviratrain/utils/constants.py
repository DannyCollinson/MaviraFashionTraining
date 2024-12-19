""" Contains constants used in the project. """

from pathlib import Path

from numpy import load as np_load
from numpy import ndarray
from numpy import save as numpy_save
from torch import load as torch_load
from torch import save as torch_save
from torchvision.io import decode_image


def np_save(array: ndarray, path: Path | str) -> None:
    """
    Provides a saving function for numpy arrays
    with a consistent interface with torch.save,
    i.e., swaps the order of the arguments

    Args:
        array (dict): NumPy array to save
        path (Path | str): path to save the array to
    """
    numpy_save(path, array)


DEFAULT_SEED = 42

# define loading functions for different image formats
LOAD_FUNCS = {
    "torchvision.io.decode_image": decode_image,
    "torch.load": torch_load,
    "np.load": np_load,
}

# define loading functions for different formats
EXTENSION_TO_LOAD_FUNC = {
    "npy": np_load,
    "np": np_load,
    "pt": torch_load,
    "pth": torch_load,
}

# define save functions for different formats
SAVE_FUNCS = {
    "np.save": np_save,
    "torch.save": torch_save,
}

# define save functions for different formats
EXTENSION_TO_SAVE_FUNC = {
    "npy": np_save,
    "np": np_save,
    "pt": torch_save,
    "pth": torch_save,
}

# define default train dataset parameters
DEFAULT_TRAIN_DATALOADER_PARAMS = {
    "batch_size": 32,
    "num_workers": 4,
    "shuffle": True,
    "drop_last": True,
    "pin_memory": True,
}

# define default val/test dataset parameters
DEFAULT_VAL_TEST_DATALOADER_PARAMS = {
    "batch_size": 32,
    "num_workers": 4,
    "shuffle": False,
    "drop_last": False,
    "pin_memory": True,
}
