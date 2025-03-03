""" Script to unzip downloaded images into data folder. """

import argparse
import os
from pathlib import Path
from subprocess import run

from maviratrain.utils.general import get_logger

# set up logger
logger = get_logger(
    "scripts.unzip_data",
    log_filename=("../logs/data_processing/data_processing.log"),
    rotation_params=(1000000, 1000),  # 1 MB, 1000 backups
)

# Create parser for command line arguments
parser = argparse.ArgumentParser(description="For parsing arguments to unzip")
parser.add_argument(
    "zipped_path",
    type=str,
    help=(
        "Path to a non-zipped directory containing zipped "
        "files or directories of files to be unzipped"
    ),
)
parser.add_argument(
    "unzip_path",
    type=str,
    help="Path to a directory to place the unzipped the files or directories",
)
parser.add_argument(
    "regex",
    type=str,
    help=(
        "Regular expression for matching files or directories "
        "in the directory specified by the zipped path"
    ),
)

if __name__ == "__main__":
    args = parser.parse_args()  # get command line arguments

    if args.zipped_path[-4:] == ".zip":
        # if zipped_path is a zip file, unzip it to temp directory
        temp_dir = Path(str(args.zipped_path[:-4]) + "-temp")
        dir_to_unzip = os.path.join(temp_dir, Path(args.zipped_path).stem)
        logger.info_(
            "Unzipping %s to temp directory %s...", args.zipped_path, temp_dir
        )
        run(["unzip", "-q", args.zipped_path, "-d", temp_dir], check=True)
    else:
        # zipped path points to a directory with zipped files
        dir_to_unzip = args.zipped_path

    logger.info_(
        "Unzipping files matching %s at %s to %s...",
        args.regex,
        dir_to_unzip,
        args.unzip_path,
    )

    # run the unzip command
    # exclude .DS_Store files macOS creates and duplicate images
    run(
        [
            "unzip",
            "-u",
            "-q",
            os.path.join(dir_to_unzip, args.regex),
            "-d",
            args.unzip_path,
            "-x",
            "**/.DS_Store",
            "**/*([0-9]).jpg",
            "**/*([0-9]).png",
        ],
        check=True,
    )

    if args.zipped_path[-4:] == ".zip":
        # clean up temp directory
        # Pylance doesn't recognize that temp_dir can't be None
        logger.info_(
            "Cleaning up temp directory %s...", temp_dir  # type: ignore
        )
        run(["rm", "-r", temp_dir], check=True)  # type: ignore

    logger.info_("Unzipping complete!")
