"""Script to run data processing pipeline"""

import argparse
from pathlib import Path

from maviratrain.data import data_processing

parser = argparse.ArgumentParser(
    description="For running data processing pipeline"
)
parser.add_argument(
    "stage",
    type=int,
    choices=[-1, 0, 1, 2, 3],
    help=(
        "Specifies after which stage the processing pipeline should stop. "
        "If -1, then the whole processing pipeline is run on the data at "
        "--raw according to any other options provided. If 0, then only file "
        "and directory name cleaning is performed on the data at --raw; all "
        "other options are ignored. If 1, then processing runs through "
        "resizing only; if 2, processing runs through train/val/test "
        "splitting; and if 3, processing is run through normalization. "
        "All unnecessary options are ignored."
    ),
)
parser.add_argument(
    "-r", "--raw", type=str, help="Path to raw data to be processed"
)
parser.add_argument(
    "--resized",
    type=str,
    help=(
        "If --raw is specified, then --resized is interpreted as the path to "
        "place data at after being resized. If --raw is not specified, then "
        "--resized is assumed to be the path to start the processing pipeline "
        "with train/val/test splitting under the assumption that the data "
        "there has already been resized. "
        "Defaults to [--raw]-resized if --raw is specified."
    ),
)
parser.add_argument(
    "--split",
    type=str,
    help=(
        "If --raw or --resized are specified, then --split is interpreted as "
        "the path to place data at after being resized. If --raw and "
        "--resized are not specified, then --split is assumed to be the path "
        "to start the processing pipeline with train/val/test splitting under "
        "the assumption that the data there has already undergone "
        "train/val/test splitting. "
        "Defaults to [--resized]-split if --resized is specified."
    ),
)
parser.add_argument(
    "-h",
    "--height",
    type=int,
    help=(
        "The height to make resized images. "
        "Ignored if not performing resizing. Defaults to 224."
    ),
    default=224,
)
parser.add_argument(
    "-w",
    "--width",
    type=int,
    help=(
        "The width to make resized images. "
        "Ignored if not performing resizing. Defaults to 224."
    ),
    default=224,
)
parser.add_argument(
    "--interp",
    type=str,
    help=(
        "Name (case insensitive) of the interpolation mode to be used by "
        "torchvision.transforms.v2.Resize. "
        "Ignored if not performing resizing. Defaults to 'bilinear'."
    ),
    default="bilinear",
)
parser.add_argument(
    "--ratios",
    type=str,
    help=(
        "String containing three numbers formatted as 'x/y/z' such that "
        "x+y+z = 100 which determine the train/val/test ratios. "
        "Defaults to '60/15/25."
    ),
    default="60/15/25",
)
parser.add_argument(
    "-c",
    "--cleanup",
    type=int,
    help=(
        "Specifies if intermediate resized dataset should be removed after "
        "splitting. Only occurs if this value is 1 and splitting is run. "
        "Defaults to 0 (no deletion)."
    ),
    default=0,
)


def run(arg: argparse.Namespace) -> dict[str, tuple[str, Path, str]]:
    """
    Runs the data processing pipeline according to the specified arguments

    Args:
        arg (argparse.Namespace): the command line arguments from argparse

    Returns:
        dict[str, tuple[str, Path, str]]: a dictionary with pipeline stage
        numbers mapped to the job_id, the result path, and whether there was
        no change or the change was in place or at a new path
    """
    # determine which stages need to be run
    stages = []
    # if -1, run whole pipeline
    if arg.stage == -1:
        assert arg.raw is not None, "If stage=-1, --raw must be specified"
        stages.extend([0, 1, 2, 3])
    # if stage is 0, run only name cleaning
    elif arg.stage == 0:
        assert arg.raw is not None, "If stage=0, --raw must be specified"
        stages.append(0)  # only run name cleaning
    # if stage is 1, the path is still specified by --raw, but no name cleaning
    elif arg.stage == 1:
        assert arg.raw is not None, "If stage=1, --raw must be specified"
        stages.append(1)  # only run resizing
    # if stage is 2, check if raw and/or resized are provided
    elif arg.stage == 2:
        assert (
            arg.raw is not None or arg.resized is not None
        ), "If stage=2, either --raw or --resized must be provided"
        if arg.raw is not None:
            stages.extend([0, 1, 2])  # run up through splitting
        else:
            stages.append(2)  # only run splitting
        # if stage is 3, check if raw, resized, or split are provided
    elif arg.stage == 3:
        assert (
            arg.raw is not None
            or arg.resized is not None
            or arg.split is not None
        ), "If stage=3, either --raw, --resized, or --split must be provided"
        if arg.raw is not None:
            stages.extend([0, 1, 2, 3])  # run the whole pipeline
        elif arg.resized is not None:
            stages.extend([2, 3])  # run starting with splitting
        else:
            stages.append(3)  # only run normalization

    # run the specified stages and store the result metadata here
    result_dict = {}

    if 0 in stages:
        # run name cleaning
        raw_path = data_processing.clean_data_naming(arg.raw)
        result_dict["0"] = ("[job_id]", Path(raw_path), "in place")  # TODO

    if 1 in stages:
        # run resizing
        resized_path = data_processing.resize_images(
            data_path=arg.raw,
            size=(arg.height, arg.width),
            interpolation=arg.interpolation,
            out_path=arg.resized,
        )
        result_dict["1"] = ("[job_id]", Path(resized_path), "new path")  # TODO

    if 2 in stages:
        # run train/val/test splitting
        ratios = tuple(arg.ratios.split("/"))
        split_path = data_processing.train_val_test(
            data_path=arg.resized,
            ratios=ratios,
            out_path=arg.split,
            seed=arg.seed,
        )
        result_dict["2"] = ("[job_id]", Path(split_path), "new path")  # TODO

    if 3 in stages:
        # run normalization
        normed_path = data_processing.normalize_data(data_path=arg.split)
        result_dict["3"] = ("[job_id]", Path(normed_path), "in place")  # TODO

    return result_dict


if __name__ == "__main__":
    args = parser.parse_args()  # get command line argument for data_path

    result = run(arg=args)  # run data processing pipeline
