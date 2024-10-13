""" Tests to see what kind of acceleration is available. """

from maviratrain.utils.general import get_device, get_logger

if __name__ == "__main__":

    # set up logger
    logger = get_logger("test_pytorch_install", level="INFO")

    get_device(verbose=True, outside_logger=logger)
