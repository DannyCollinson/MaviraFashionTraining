"""Common utility functions"""

import datetime
import os

from torch.cuda import is_available as is_cuda_available
from torch.backends.mps import is_available as is_mps_available


def get_log_time() -> str:
    """
    Returns the current UTC datetime in ISO format truncated to seconds

    Returns:
        str -- the current datetime
    """
    return datetime.datetime.now(tz=datetime.timezone.utc).isoformat(
        timespec="seconds"
    )


def get_file_date() -> str:
    """
    Returns the current UTC date in ISO format

    Returns:
        str -- the current date
    """
    return datetime.datetime.now(tz=datetime.timezone.utc).isoformat(
        timespec="seconds"
    )[:10]


def is_valid_directory(data_path) -> bool:
    """
    Verifies that data_path points to a directory

    Arguments:
        data_path {_type_} -- path to check for directory

    Returns:
        bool -- True if directory exists, false otherwise
    """
    # make sure data directory exists
    res = os.path.isdir(data_path)
    if not res:
        print(f"Expected data_path to point to a directory. Got {data_path}.")
    return res


def get_device(verbose: bool = False) -> str:
    """
    Returns the string of the available accelerator

    Returns:
        str: device string corresponding to available accelerator
    """
    if is_cuda_available():
        if verbose:
            print("CUDA available")
        return "cuda"
    if verbose:
        print("CUDA not available")

    if is_mps_available():
        if verbose:
            print("MPS available")
        return "mps"
    if verbose:
        print("MPS not available")

    if verbose:
        print("Only CPU available")
    return "cpu"
