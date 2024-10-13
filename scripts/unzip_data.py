""" Script to unzip downloaded images into data folder. """

import argparse
import os

from maviratrain.utils.general import get_logger

# set up logger
logger = get_logger(
    "scripts.unzip_data",
    # should be running from a notebook, hence the ../
    log_filename="../logs/data_processing/data_processing.log",
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

    # have to escape the * for the command to work
    if args.regex == "*.zip":
        args.regex = r"\*.zip"

    logger.info_(
        "Unzipping files matching %s at %s to %s...",
        args.regex,
        args.zipped_path,
        args.unzip_path,
    )

    # run the unzip command
    os.system(
        f"unzip -u "
        f"{os.path.join(args.zipped_path, args.regex)} -d {args.unzip_path}"
    )

    logger.info_("Unzipping complete!")
