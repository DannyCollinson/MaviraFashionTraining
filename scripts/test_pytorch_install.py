""" Tests to see what kind of acceleration is available. """

from numpy import dot
from torch import manual_seed as torch_set_seed
from torch import matmul, randn  # pylint: disable=no-name-in-module

from maviratrain.utils.constants import DEFAULT_SEED
from maviratrain.utils.general import get_device, get_logger

if __name__ == "__main__":

    # set up logger
    logger = get_logger("test_pytorch_install", level="INFO")

    # test which device is available to PyTorch
    DEVICE = get_device(verbose=True, outside_logger=logger)

    # set seed for reproducibile tests
    torch_set_seed(DEFAULT_SEED)

    # get random test matrices
    test_matrix_a = randn(4, 3, device="cpu")
    test_matrix_b = randn(3, 5, device="cpu")

    # use numpy
    numpy_matrix_a = test_matrix_a.numpy()
    numpy_matrix_b = test_matrix_b.numpy()
    numpy_result = dot(numpy_matrix_a, numpy_matrix_b)

    # use torch
    torch_tensor_a = test_matrix_a.to(DEVICE)
    torch_tensor_b = test_matrix_b.to(DEVICE)
    torch_result = matmul(torch_tensor_a, torch_tensor_b)

    # round to 6 decimal places
    np_result = numpy_result.round(6)  # pylint: disable=no-member
    torch_result_np = torch_result.cpu().numpy().round(6)

    # compare results
    assert (
        np_result == torch_result_np
    ).all(), "NumPy and PyTorch results do not match!"
