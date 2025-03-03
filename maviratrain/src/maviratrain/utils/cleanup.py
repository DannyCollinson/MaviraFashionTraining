"""
Utility functions for resetting/clearing logs, checkpoints, etc.
"""

from pathlib import Path
from subprocess import run

from .general import get_logger

# set up logger
logger = get_logger("mt.utils.cleanup")


def clean_up_intermediate_dataset(data_path: Path | str) -> None:
    """
    Deletes intermediate dataset directories
    created during data processing.

    Args:
        data_path (str): path to the intermediate dataset directory
    """
    logger.set_log_filename("../logs/data_processing/data_processing.log")
    logger.debug_("Deleting intermediate dataset at %s...", data_path)

    # delete the intermediate dataset directory
    run(["rm", "-r", data_path], check=True)

    logger.debug_("Removed intermediate dataset!")


def clean_up_checkpoint(checkpoint_path: Path | str) -> None:
    """
    Deletes the given checkpoint.

    Args:
        checkpoint_path (str): path to the checkpoint
    """
    logger.set_log_filename("../logs/train_runs/classifier/training.log")
    logger.debug_("Deleting checkpoint at %s", checkpoint_path)

    # delete the checkpoint
    run(["rm", checkpoint_path], check=True)


def clean_up_checkpoints(checkpoint_dir: Path | str) -> None:
    """
    Deletes all stale checkpoints in the given directory.

    Args:
        checkpoint_dir (str): path to the directory
            containing checkpoints to be cleaned up
    """
    logger.set_log_filename("../logs/train_runs/classifier/training.log")
    logger.debug_("Deleting stale checkpoints in %s", checkpoint_dir)

    # TODO: implement this function
