""" Script to run data processing pipeline. """

import argparse
from pathlib import Path
from subprocess import run as subprocess_run
from typing import Any

from maviratrain.data import data_processing
from maviratrain.utils.cleanup import clean_up_intermediate_dataset
from maviratrain.utils.general import get_logger, get_time
from maviratrain.utils.registration.register_data import (
    get_dataset_id,
    register_processing_job,
    update_processing_job,
)

# set up logger
logger = get_logger(
    "scripts.run_data_processing",
    # should be running from a notebook, hence the ../
    log_filename="../logs/data_processing/data_processing.log",
    rotation_params=(1000000, 1000),  # 1 MB, 1000 backups
)

# set up argument parser
parser = argparse.ArgumentParser(
    description="Run the data processing pipeline"
)
parser.add_argument(
    "stage",
    type=int,
    choices=[-1, 0, 1, 2, 3, 4],
    help=(
        "Specifies after which stage the processing pipeline"
        "should stop. If -1, then the whole processing pipeline is run "
        "on the data at --raw according to any other options provided. "
        "If 0, then only file and directory name cleaning is performed "
        "on the data at --raw; all other options are ignored. "
        "If 1, then processing runs through resizing only; "
        "if 2, processing runs through train/val/test splitting; "
        "if 3, processing is run through normalization; and "
        "if 4, processing runs through file format conversion. "
        "All unnecessary options are ignored."
    ),
)
parser.add_argument(
    "-r",
    "--raw",
    type=str,
    help="String. Path to raw data to be processed.",
    default=None,
)
parser.add_argument(
    "--resized",
    type=str,
    help=(
        "String. If --raw is specified, then --resized is interpreted "
        "as the path to place data at after being resized. "
        "If only running resizing, then the data at --raw is assumed "
        "to be data for resizing that has already been cleaned. "
        "If --raw is not specified, then --resized is assumed to be "
        "the path to start the processing pipeline "
        "with train/val/test splitting under the assumption that "
        "the data there has already been resized. "
        "Defaults to [--raw]-r[job_id] if --raw is specified."
    ),
    default=None,
)
parser.add_argument(
    "--split",
    type=str,
    help=(
        "String. If --raw or --resized are specified, then --split is "
        "interpreted as the path to place data at after being resized. "
        "If --raw and --resized are not specified, "
        "then --split is assumed to be the path "
        "to start the processing pipeline with normalization "
        "under the assumption that the data there "
        "has already undergone train/val/test splitting. "
        "If stage is 3, then --split is the path "
        "to already split data that is to be normalized. "
        "If stage is 4, then --split is the path to the data to be "
        "normalized and converted to a new format. "
        "Defaults to [--resized]-s[job_id] if --resized is specified."
    ),
    default=None,
)
parser.add_argument(
    "--normalized",
    type=str,
    help=(
        "String. If --raw or --resized or --split are specified, "
        "then --normalized is interpreted as the path to place data at "
        "after being normalized. If stage is equal to 4, then "
        "--normalized is the path to the data to be converted. "
        "Defaults to [--split]-n[job_id] if --split is specified."
    ),
    default=None,
)
parser.add_argument(
    "--converted",
    type=str,
    help=(
        "String. If --raw, --resized, --split, or --normalized are "
        "specified, then --converted is interpreted as the path to "
        "place data at after being converted to a new format. Defaults "
        "to [--normalized]-c[job_id] if --normalized is specified."
    ),
    default=None,
)
parser.add_argument(
    "--height",
    type=int,
    help=(
        "Integer. The height to make resized images. "
        "Ignored if not performing resizing. Defaults to 224."
    ),
    default=224,
)
parser.add_argument(
    "--width",
    type=int,
    help=(
        "Integer. The width to make resized images. "
        "Ignored if not performing resizing. Defaults to 224."
    ),
    default=224,
)
parser.add_argument(
    "--interp",
    type=str,
    help=(
        "String. Name (case insensitive) of the interpolation mode "
        "to be used by torchvision.transforms.v2.Resize. "
        "Ignored if not performing resizing. Defaults to 'bilinear'."
    ),
    default="bilinear",
)
parser.add_argument(
    "--deduplicate",
    type=int,
    help=(
        "Integer; 0 or 1. Specifies if duplicate images should be "
        "removed during resizing. Defaults to 1 (remove duplicates)."
    ),
    default=1,
)
parser.add_argument(
    "--ratios",
    type=str,
    help=(
        "String. String containing three numbers formatted "
        "as 'x/y/z' such that x+y+z = 100. "
        "Determines the train/val/test ratios. "
        "Defaults to '64/16/20'."
    ),
    default="64/16/20",
)
parser.add_argument(
    "--seed",
    type=int,
    help=(
        "Integer. Seed to use for python's random number generation. "
        "Defaults to 42."
    ),
    default=42,
)
parser.add_argument(
    "--norm_method",
    type=str,
    help=(
        "String. Name of the normalization method to use. "
        "Current options are "
        "'zscore', 'pixelz', 'localz', 'minmax', 'minmaxextended', "
        "'localminmax', and localminmaxextended. "
        "Ignored if not performing normalization. "
        "Defaults to 'zscore'."
    ),
    default="zscore",
)
parser.add_argument(
    "--final_format",
    type=str,
    help=(
        "String. Format to convert images to, given as a filename "
        "extension. Ignored if not performing conversion. "
        "If the same as the working format, then no changes are made. "
        "Defaults to 'npy', for numpy array."
    ),
    default="npy",
)
parser.add_argument(
    "--quality",
    type=int,
    help=(
        "Integer. Quality of conversion to JPEG. Between 0 and 100. "
        "Ignored if not performing JPEG conversion. Defaults to 95."
    ),
    default=95,
)
parser.add_argument(
    "--working_format",
    type=str,
    help=(
        "String. Format that the data will be saved as after "
        "resizing and subsequent intermediate stages, given as a "
        "filename extension. Ignored if only performing conversion."
        "Defaults to 'npy', for numpy array."
    ),
    default="npy",
)
parser.add_argument(
    "-c",
    "--cleanup",
    type=int,
    help=(
        "Integer; 0 or 1. Specifies if intermediate resized dataset "
        "should be removed after splitting. "
        "Only occurs if this value is 1 and splitting is run. "
        "Defaults to 0 (no deletion)."
    ),
    default=0,
)
parser.add_argument(
    "--notes",
    type=str,
    help=(
        "String. Any notes to include in the database entry for the "
        "data processing job. Defaults to None."
    ),
    default=None,
)


def get_stages(arg: argparse.Namespace) -> list[int]:
    """
    Get the stages to run based on the command line arguments

    Args:
        arg (argparse.Namespace): the command line arguments
            from argparse

    Returns:
        list[int]: list representing the stages to run
    """
    logger.debug_(
        "Parsing command line arguments to determine stages to run..."
    )

    # store list of integers representing stages to run
    stages = []

    # if -1, run whole pipeline
    if arg.stage == -1:
        assert (
            arg.raw is not None and arg.raw != "None"
        ), "If stage=-1, --raw must be specified"
        stages.extend([0, 1, 2, 3, 4])

    # if stage is 0, run only name cleaning
    elif arg.stage == 0:
        assert (
            arg.raw is not None and arg.raw != "None"
        ), "If stage=0, --raw must be specified"
        stages.append(0)  # only run name cleaning

    # if stage is 1, the path is still specified by --raw, but no name cleaning
    elif arg.stage == 1:
        assert (
            arg.raw is not None and arg.raw != "None"
        ), "If stage=1, --raw must be specified"
        stages.append(1)  # only run resizing

    # if stage is 2, check if raw and/or resized are provided
    elif arg.stage == 2:
        assert (arg.raw is not None and arg.raw != "None") or (
            arg.resized is not None and arg.resized != "None"
        ), "If stage=2, either --raw or --resized must be provided"
        if arg.raw is not None and arg.raw != "None":
            stages.extend([0, 1, 2])  # run up through splitting
        else:
            stages.append(2)  # only run splitting

    # if stage is 3, check if raw, resized, or split are provided
    elif arg.stage == 3:
        assert (
            (arg.raw is not None and arg.raw != "None")
            or (arg.resized is not None and arg.resized != "None")
            or (arg.split is not None and arg.split != "None")
        ), "If stage=3, either --raw, --resized, or --split must be provided"
        if arg.raw is not None and arg.raw != "None":
            stages.extend([0, 1, 2, 3])  # run from beginning
        elif arg.resized is not None and arg.resized != "None":
            stages.extend([2, 3])  # run starting with splitting
        else:
            stages.append(3)  # only run normalization

    # if stage is 4, check all possible paths
    elif arg.stage == 4:
        assert (
            (arg.raw is not None and arg.raw != "None")
            or (arg.resized is not None and arg.resized != "None")
            or (arg.split is not None and arg.split != "None")
            or (arg.normalized is not None and arg.normalized != "None")
        ), (
            "If stage=4, either --raw, --resized, "
            "--split, or --normalized must be provided"
        )
        if arg.raw is not None and arg.raw != "None":
            stages.extend([0, 1, 2, 3, 4])  # run from beginning
        elif arg.resized is not None and arg.resized != "None":
            stages.extend([2, 3, 4])  # run starting with splitting
        elif arg.split is not None and arg.split != "None":
            stages.extend([3, 4])  # run starting with normalization
        else:
            stages.extend([4])  # run only conversion

    logger.debug_("Determined stages to run: %s", stages)
    return stages


def run(arg: argparse.Namespace) -> dict[str, Any]:
    """
    Runs the data processing pipeline
    according to the specified arguments

    Args:
        arg (argparse.Namespace): the command line arguments
            from argparse

    Returns:
        dict[str, Any]: a dictionary with all of the information
            about the data processing job
    """
    # store results in a dictionary
    result: dict[str, Any] = {}
    preregistration_dict: dict[str, Any] = {}

    # get job ID based on the previous job ID
    job_id = data_processing.get_data_processing_job_id(new=True)
    result["job_id"] = job_id

    logger.info_("Running data processing pipeline with job ID %s...", job_id)

    # get start time to store in database
    result["start_time"] = get_time()
    preregistration_dict["start_time"] = result["start_time"]

    # set end time to None for preregistration
    preregistration_dict["end_time"] = None

    # determine which stages need to be run
    stages = get_stages(arg)
    result["stages"] = stages
    preregistration_dict["stages"] = stages

    # store the raw data path in the result dictionary
    result["raw_path"] = arg.raw

    # try to get the starting dataset id from the database using the raw path
    if arg.raw is not None and arg.raw != "None":
        try:
            result["starting_dataset_id"] = get_dataset_id(data_path=arg.raw)
        except RuntimeError as e:
            result["starting_dataset_id"] = None
            logger.error_(str(e))
    else:
        result["starting_dataset_id"] = None

    # determine the starting dataset path based on the stages to run
    if 0 in stages:
        result["starting_dataset_path"] = arg.raw
    elif 1 in stages:
        result["starting_dataset_path"] = arg.raw
    elif 2 in stages:
        result["starting_dataset_path"] = arg.resized
    elif 3 in stages:
        result["starting_dataset_path"] = arg.split
    elif 4 in stages:
        result["starting_dataset_path"] = arg.normalized
    preregistration_dict["starting_dataset_path"] = result[
        "starting_dataset_path"
    ]

    # add run metadata to the preregistration dictionary
    preregistration_dict["0"] = None
    preregistration_dict["1"] = None
    preregistration_dict["2"] = None
    preregistration_dict["3"] = None
    preregistration_dict["4"] = None
    preregistration_dict["resize_height"] = arg.height
    preregistration_dict["resize_width"] = arg.width
    preregistration_dict["interpolation"] = arg.interp
    preregistration_dict["deduplication"] = bool(arg.deduplicate)
    preregistration_dict["cleanup"] = bool(arg.cleanup)
    preregistration_dict["ratios"] = [
        int(num) for num in arg.ratios.split("/")
    ]
    preregistration_dict["stats_path"] = None
    preregistration_dict["norm_method"] = arg.norm_method
    preregistration_dict["conversion_format"] = arg.final_format
    preregistration_dict["jpeg_quality"] = arg.quality
    preregistration_dict["seed"] = arg.seed

    # register the job in the postgres database
    preregistration_job_id = register_processing_job(
        result_dict=preregistration_dict, notes=arg.notes
    )

    # log cleanup option in result dictionary
    if arg.cleanup:
        result["cleanup"] = True
    else:
        result["cleanup"] = False

    # run the specified stages
    if 0 in stages:  # name cleaning
        result["starting_dataset_path"] = arg.raw
        # run name cleaning
        renamed_id, renamed_path = data_processing.clean_data_naming(
            data_path=arg.raw, job_id=job_id
        )
        result["0"] = (renamed_id, Path(renamed_path))
    else:
        # if not cleaning, set renamed_path to None for next stages
        renamed_path = None
        result["0"] = None

    if 1 in stages:  # resizing
        # determine which input path to use for resizing
        if renamed_path:
            use_path = renamed_path
        else:
            use_path = arg.raw
            result["starting_dataset_path"] = arg.raw
        # run resizing
        resized_id, resized_path = data_processing.resize_images(
            data_path=use_path,
            size=(arg.height, arg.width),
            interpolation=arg.interp,
            save_format=arg.working_format,
            deduplicate=arg.deduplicate,
            out_path=arg.resized,
        )
        result["resize_height"] = arg.height
        result["resize_width"] = arg.width
        result["interpolation"] = arg.interp
        result["1"] = (resized_id, Path(resized_path))
    else:
        resized_path = None  # set to None for next stages
        # if not resizing, set interpolation to None for registering job
        result["interpolation"] = None
        result["1"] = None

    if 2 in stages:  # train/val/test splitting
        # determine which input path to use for splitting
        if resized_path:
            use_path = resized_path
        else:
            use_path = arg.resized
            result["starting_dataset_path"] = arg.resized
        # run train/val/test splitting
        ratios = [int(num) for num in arg.ratios.split("/")]
        split_id, split_path = data_processing.train_val_test(
            data_path=use_path,
            ratios=ratios,  # type: ignore  # list instead of tuple
            out_path=arg.split,
            seed=arg.seed,
        )
        result["ratios"] = ratios
        result["seed"] = arg.seed
        result["2"] = (split_id, Path(split_path))
    else:
        split_path = None  # set to None for next stages
        # if not splitting, set ratios and seed to None for registering job
        result["ratios"] = [None, None, None]
        result["seed"] = None
        result["2"] = None

    # clean up intermediate dataset if resizing and splitting were run
    if arg.cleanup and resized_path and split_path:
        clean_up_intermediate_dataset(data_path=resized_path)

    if 3 in stages:  # normalization
        # determine which input path to use for normalization
        if split_path:
            use_path = split_path
        else:
            use_path = arg.split
            result["starting_dataset_path"] = arg.split
        # run normalization
        _, normed_id, normed_path, stats_path = data_processing.normalize_data(
            data_path=use_path, method=arg.norm_method, out_path=arg.normalized
        )
        result["norm_method"] = arg.norm_method
        result["3"] = (normed_id, Path(normed_path))
        result["stats_path"] = stats_path
    else:
        normed_path = None  # set to None for next stages
        # if not normalizing, set attributes to None for registering job
        result["norm_method"] = None
        result["3"] = None
        result["stats_path"] = None

    # clean up intermediate dataset if splitting and normalization were run
    if arg.cleanup and split_path and normed_path:
        clean_up_intermediate_dataset(data_path=split_path)

    if 4 in stages:  # conversion to new file format
        # skip conversion if the working format is the same as the final format
        if 1 in stages and arg.working_format == arg.final_format:
            converted_path = None
            result["4"] = None
            result["jpeg_quality"] = None
            logger.info_(
                "Skipping conversion because the working format "
                "is the same as the final format."
            )
        else:
            # determine which input path to use for conversion
            if normed_path:
                use_path = normed_path
            else:
                use_path = arg.normalized
                result["starting_dataset_path"] = arg.normalized
            # run conversion to new file format
            if arg.final_format == "npy":
                _, converted_dataset_id, converted_path = (
                    data_processing.convert_to_numpy(
                        data_path=use_path, out_path=arg.converted
                    )
                )
            elif arg.final_format == "pt":
                _, converted_dataset_id, converted_path = (
                    data_processing.convert_to_pytorch(
                        data_path=use_path, out_path=arg.converted
                    )
                )
            elif arg.final_format == "jpeg":
                _, converted_dataset_id, converted_path = (
                    data_processing.convert_to_jpeg(
                        data_path=use_path,
                        quality=arg.quality,
                        out_path=arg.converted,
                    )
                )
                result["jpeg_quality"] = arg.quality
            else:
                raise ValueError(
                    f"Unsupported final format: {arg.final_format}"
                )
            result["4"] = (converted_dataset_id, Path(converted_path))
    else:
        converted_path = None  # set to None for consistency
        # if not converting, set quality to None for registering job
        result["jpeg_quality"] = None
        result["4"] = None

    # clean up intermediate dataset if normalization and conversion were run
    if arg.cleanup and normed_path and converted_path:
        clean_up_intermediate_dataset(data_path=normed_path)

    # get end time to store in database
    result["end_time"] = get_time()

    # update the preregistered job with the final information
    returned_job_id = update_processing_job(
        result_dict=result, job_id=preregistration_job_id
    )

    # check if the job ID returned from the database matches the one generated
    assert returned_job_id == job_id, "Database returned unexpected job ID"

    # zip up logs to save space
    log_archive_path = f"../logs/archive/data_processing/j{job_id}"
    logger.info_("Logs zipped and moved to %s", log_archive_path + ".zip")

    logger.info_("Finished running job %s!", job_id)
    return result


if __name__ == "__main__":
    logger.debug_("Running run_data_processing.py...")

    # get command line argument for data_path
    args = parser.parse_args()

    # run data processing pipeline
    result_dict = run(arg=args)

    logger.debug_("Finished runnning run_data_processing.py!")

    # zip up logs to save space
    logs_path = Path("../logs/data_processing")
    log_archive = f"../logs/archive/data_processing/j{result_dict['job_id']}"
    logger.close_logger()  # close logger to avoid file conflicts

    # move logs to job-specific directory
    for file in logs_path.glob("*"):
        subprocess_run(["mv", file, log_archive], check=True)

    # zip logs
    subprocess_run(
        ["zip", "-9", log_archive + ".zip", log_archive], check=True
    )

    # remove unzipped logs
    subprocess_run(["rm", "-r", log_archive], check=True)
