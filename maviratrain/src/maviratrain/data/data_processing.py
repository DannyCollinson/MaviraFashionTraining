"""
Procedures for cleaning and processing data
and generating train/val/test splits.
"""

import os
import random
import re
from pathlib import Path

import torch
from PIL import Image
from psycopg import connect
from torchvision.io import ImageReadMode, decode_image
from torchvision.transforms.v2 import InterpolationMode, Resize

from ..utils.general import (
    get_data_processing_job_id,
    get_logger,
    get_postgres_connection_string,
    is_valid_dataset,
)
from ..utils.registration.register_data import register_dataset

# set up logger
logger = get_logger(
    "mt.data.data_processing",
    # should be running from a notebook, hence the ../
    log_filename="../logs/data_processing/data_processing.log",
    rotation_params=(1000000, 1000),  # 1 MB, 1000 backups
)


def clean_subdirectories(data_path: Path | str) -> Path:
    """
    Renames subdirectories under data_path to remove '_files' suffix,
    making sure this would not cause duplicates

    Args:
        data_path (Path | str): path to top of data directory to clean

    Raises:
        FileExistsError: if a rename would cause a duplicate

    Returns:
        Path: the original data_path
    """
    logger.debug_("Cleaning subdirectory names at %s...", data_path)

    assert is_valid_dataset(data_path=data_path, outside_logger=logger)

    for cur_dir, sub_dirs, _ in os.walk(data_path):
        for sub_dir in sub_dirs:
            if "_files" in sub_dir:
                # check for duplicate creation
                if sub_dir.removesuffix("_files") in sub_dirs:
                    message = (
                        f"Renaming {sub_dir} would result "
                        "in two directories named "
                        f"{sub_dir.removesuffix('_files')}. "
                        f"Please change the name of {sub_dir} "
                        "manually and rerun the script."
                    )
                    logger.error_("%s", message)
                    raise FileExistsError(message)

                subdir_path = os.path.join(cur_dir, sub_dir)
                os.rename(subdir_path, subdir_path.removesuffix("_files"))
                # log directory name change
                logger.debug_(
                    "%s renamed to %s",
                    subdir_path,
                    subdir_path.removesuffix("_files"),
                )
    logger.debug_("Done cleaning subdirectory names!")
    return Path(data_path)


def clean_filenames(data_path: Path | str) -> Path:
    """
    Renames raw data files to have consistent formatting: im#####.jpg

    Args:
        data_path (Path | str): path to top of data directory to clean

    Raises:
        ValueError: if filename is not of the expected form

    Returns:
        Path: the original data_path
    """
    logger.debug_("Cleaning filenames at %s...", data_path)

    assert is_valid_dataset(data_path=data_path, outside_logger=logger)

    for cur_dir, _, files in os.walk(data_path):
        for index, filename in enumerate(files):
            # check if filename has already been cleaned
            if re.match(r"im\d{5}\.jpg", filename):
                logger.debug_("%s already cleaned", filename)
                continue

            if "images" not in filename or (
                "jpg" not in filename and "png" not in filename
            ):
                message = (
                    f"Filename {filename} was not expected. "
                    "Was expecting filename of the form "
                    "images.jpg, images.png, images_###.jpg, "
                    "images_###.png, images_####.jpg, or images_####.png"
                )
                logger.error_("%s", message)
                raise ValueError(message)

            # define new filename with im#####.jpg scheme
            new_filename = (
                "im" + "0" * (5 - len(str(index))) + str(index) + ".jpg"
            )

            # handle special case of PNGs (RGB+Alpha ?) by loading as RGB,
            # deleting PNG file, and resaving under a new JPG filename
            if filename.split(".")[-1] == "png":
                imfile = Image.open(
                    os.path.join(cur_dir, filename), formats=["PNG"]
                )
                im = imfile.convert("RGB")
                im.save(os.path.join(cur_dir, new_filename), format="JPEG")
                logger.debug_("%s saved as %s", filename, new_filename)
                os.remove(os.path.join(cur_dir, filename))
                logger.debug_("%s removed", filename)
            else:
                # otherwise rename file with new filename
                old_filepath = os.path.join(cur_dir, filename)
                new_filepath = os.path.join(cur_dir, new_filename)
                os.rename(old_filepath, new_filepath)
                logger.debug_("%s renamed as %s", old_filepath, new_filepath)
    logger.debug_("Done cleaning filenames!")
    return Path(data_path)


def clean_data_naming(
    data_path: Path | str, job_id: int | None = None
) -> tuple[int, Path]:
    """
    Cleans the subdirectory names (remove '_files' suffix)

    Args:
        data_path (Path | str): path to top of data directory to clean

    Returns:
        Path: the original data_path
    """
    logger.info_("Cleaning data naming at %s...", data_path)

    assert is_valid_dataset(data_path=data_path, outside_logger=logger)

    clean_subdirectories(data_path=data_path)
    clean_filenames(data_path=data_path)

    # register the newly cleaned dataset to the database
    dataset_id = register_dataset(
        data_path=data_path, notes="Cleaned up naming"
    )

    logger.info_("Done cleaning data names!")
    return dataset_id, Path(data_path)


def resize_images(
    data_path: Path | str,
    size: tuple[int, int] = (224, 224),
    interpolation: str = "bilinear",
    out_path: Path | str | None = None,
) -> tuple[int, Path]:
    """
    Resizes images in data_path to size and saves them in out_path

    Args:
        data_path (Path | str): path to top of data directory to resize
        size (tuple[int, int], optional): dimensions to resize
            images to. Defaults to (224, 224).
        interpolation (str, optional): torchvision interpolation mode to
            use for resizing images. Defaults to "bilinear".
        out_path (Path | str | None, optional): path to save resized
            images out to. Defaults to None.

    Raises:
        NotImplementedError: if out_path already exists
        ValueError: if interpolation mode is not supported

    Returns:
        tuple[int, Path]: the dataset ID and the path to the resized data
    """
    # specify default out_path if not given
    job_id = get_data_processing_job_id()
    out_path = Path(str(data_path).removesuffix("/") + "-r" + str(job_id))

    logger.info_(
        "Resizing images at %s to size %s and saving at %s...",
        data_path,
        size,
        out_path,
    )

    assert is_valid_dataset(data_path=data_path, outside_logger=logger)

    # make sure destination doesn't already exist to avoid issues
    if os.path.exists(out_path):
        message = (
            f"Destination subdirectory {out_path} "
            "already exists, which is currently not handled"
        )
        logger.error_("%s", message)
        raise NotImplementedError(message)

    os.mkdir(out_path)
    logger.debug_("Created output directory %s", out_path)

    # get the correct interpolation mode
    interpolation = interpolation.upper()  # for case insensitivity
    interpolation_mode = InterpolationMode.BILINEAR
    if interpolation == "BILINEAR":
        interpolation_mode = InterpolationMode.BILINEAR
    elif interpolation == "BICUBIC":
        interpolation_mode = InterpolationMode.BICUBIC
    elif interpolation == "NEAREST":
        interpolation_mode = InterpolationMode.NEAREST
    elif interpolation == "NEAREST_EXACT":
        interpolation_mode = InterpolationMode.NEAREST_EXACT
    else:
        message = (
            f"Provided interpolation mode '{interpolation}' is not supported. "
            "Supported Torchvision InterpolationModes include "
            "BILINEAR, BICUBIC, NEAREST, and NEAREST_EXACT."
        )
        logger.error_("%s", message)
        raise ValueError(message)

    # define transformation
    resize = Resize(
        size=size, interpolation=interpolation_mode, antialias=True
    )

    for cur_dir, sub_dirs, files in os.walk(data_path):
        # replicate the data_path subdirectory structure
        for sub_dir in sub_dirs:
            destination_subdir = os.path.join(out_path, sub_dir)
            os.mkdir(destination_subdir)
            logger.debug_("Created destination %s", destination_subdir)

        # resize each file and place in out_path/sub_dir/fname.pt
        for filename in files:
            filepath = os.path.join(cur_dir, filename)
            im = decode_image(filepath, mode=ImageReadMode.RGB)
            # conver to float before resizing for accuracy
            im = im.to(dtype=torch.float32)
            im = resize(im)
            # get image subpath by removing data_path from filepath
            file_subpath = filepath.removeprefix(str(data_path) + "/")
            file_subpath = file_subpath.removeprefix(str(data_path))
            # create new filepath as outpath/subpath
            new_filepath = os.path.join(out_path, file_subpath)
            new_filepath = new_filepath.replace(".jpg", ".pt")
            torch.save(im, new_filepath)
            logger.debug_("Resized %s and saved as %s", filepath, new_filepath)

    # register the newly created dataset to the database
    dataset_id = register_dataset(data_path=out_path, notes="Resized images")

    logger.info_("Done resizing images!")
    return dataset_id, Path(out_path)


def train_val_test(
    data_path: Path | str,
    ratios: tuple[int, int, int] = (60, 15, 25),
    out_path: Path | str | None = None,
    seed: int = 42,
) -> tuple[int, Path]:
    """
    Randomly places files in data_path into train, val, and test files
    at the out_path according to the ratios in 'ratios'.

    Args:
        data_path (Path | str): path to the
            unsplit and unnormalized data
        ratios (tuple[int, int, int], optional): percentages of each
            class to split into train/val/test. Each must be >= 0
            and <= 100 and must have sum=100, e.g., (70, 10, 20) for 70%
            train, 10% val, 20% test. Defaults to (60, 15, 25).
        out_path (Path | str | None, optional): path where data
            should be split into. Defaults to data_path + '-s[job_id]'.
        seed (int, optional): seed for the python built-in random
            generator. Defaults to 42.

    Returns:
        tuple[int, Path]: the dataset ID and the path to the split data
    """
    # specify default out_path if not given
    job_id = get_data_processing_job_id()
    out_path = Path(str(data_path).removesuffix("/") + "-s" + str(job_id))

    logger.info_(
        "Splitting data at %s into train/val/test sets at %s "
        "using ratios %s and random seed %s...",
        data_path,
        out_path,
        ratios,
        seed,
    )

    assert is_valid_dataset(data_path=data_path, outside_logger=logger)

    # ensure valid ratios input
    assert (
        len(ratios) == 3
        and all(ratios) >= 0
        and all(ratios) <= 100
        and sum(ratios) == 100
    ), (
        "Expected all 3 numbers in ratios to be >= 0 and <= 100 "
        f"and for sum of all numbers to be 100. Got {ratios}."
    )

    # make sure destination doesn't already exist to avoid issues
    if os.path.exists(out_path):
        message = (
            f"Destination subdirectory {out_path} "
            "already exists, which is currently not handled"
        )
        logger.error_("%s", message)
        raise NotImplementedError(message)

    os.mkdir(out_path)
    logger.debug_("Created output directory %s", out_path)

    # set random seed for reproducibility
    random.seed(seed)

    # create directories for splits
    splits = ["train", "val", "test"]
    for split in splits:
        split_dir = os.path.join(out_path, split)
        os.mkdir(split_dir)
        logger.debug_("Created output directory %s", split_dir)

    for cur_dir, sub_dirs, files in os.walk(data_path):
        # replicate data_path subdirectory format in destination
        for sub_dir in sub_dirs:
            for split in splits:
                new_dir = os.path.join(out_path, split, sub_dir)
                os.mkdir(new_dir)
                logger.debug_("Created output directory %s", new_dir)
        # skip forward if there are no files to sort in current directory
        if len(files) == 0:
            continue
        # get random permution of files
        shuffled_files = random.sample(files, k=len(files))
        # get current subdirectory to replicate within each split directory
        cur_subdir = cur_dir.removeprefix(str(data_path) + "/")
        cur_subdir = cur_subdir.removeprefix(str(data_path))
        split_dirs = [
            os.path.join(out_path, split, cur_subdir) for split in splits
        ]
        # calculate number of samples in subdir to go in each split
        split_nums = [
            len(files) * ratios[0] // 100,  # train
            len(files) * ratios[1] // 100,  # val
            len(files)
            - len(files) * ratios[0] // 100  # test
            - len(files) * ratios[1] // 100,
        ]
        # sort files into correct folders
        for i, split_dir in enumerate(split_dirs):
            logger.debug_("Copying %s files to %s", split_nums[i], split_dir)
            for _ in range(int(split_nums[i])):
                # draw total of len(files) files from end of file list
                filename = shuffled_files.pop()
                # get current and new filepaths
                filepath = os.path.join(cur_dir, filename)
                new_filepath = os.path.join(split_dir, filename)
                # copy to new filepath
                os.system(f"cp {filepath} {new_filepath}")
                logger.debug_("Copied %s to %s", filepath, new_filepath)

    # register the newly created dataset to the database
    dataset_id = register_dataset(
        data_path=out_path, notes="Split into train/val/test"
    )

    logger.info_("Done creating train/val/test splits!")
    return dataset_id, Path(out_path)


# def collect_stats(
#     data_path: Path | str,
# ) -> tuple[Tensor, Tensor]:
#     """
#     Collects mean and standard deviations for the data

#     Args:
#         data_path (Path | str): path to top of data directory to
#             collect stats

#     Returns:
#         tuple[Tensor, Tensor]: the mean and standard deviations
#             for all three channels
#     """
#     logger.debug_("Collecting statistics for data at %s...", data_path)

#     assert is_valid_dataset(data_path=data_path, outside_logger=logger)

#     # initialize stats for tracking
#     num_files = 0
#     channel_means = torch.tensor([0.0, 0.0, 0.0])
#     channel_stds = torch.tensor([0.0, 0.0, 0.0])

#     # collect stats from training files
#     for cur_dir, _, files in os.walk(os.path.join(data_path)):
#         for file in files:
#             filepath = os.path.join(cur_dir, file)
#             im = torch.load(filepath, map_location="cpu", weights_only=True)
#             num_files += 1
#             channel_means += im.mean(dim=[1, 2])
#             channel_stds += im.std(dim=[1, 2])
#     channel_means /= num_files
#     channel_stds /= num_files

#     logger.debug_("Done collecting statistics!")
#     return channel_means, channel_stds


# def normalize_tensor(
#     tensor: Tensor,
#     means: Tensor,
#     stds: Tensor,
# ) -> Tensor:
#     """
#     Normalizes tensor by subtracting means
#     and dividing by the standard deviations

#     Args:
#         tensor (Tensor): tensor to normalize
#         means (Tensor): means for each channel
#         stds (Tensor): standard deviations for each channel

#     Returns:
#         Tensor: the normalized tensor
#     """
#     means = means[:, None, None]
#     stds = stds[:, None, None]
#     return (tensor - means) / stds


# def normalize_data(
#     data_path: Path | str, method: str
# ) -> tuple[int, Path, Path]:
#     """
#     Normalizes data in train, val, and test directories at data_path
#     according to statistics of the train data by subtracting per-channel
#     means and dividing by per-channel standard deviations

#     Args:
#         data_path (Path | str): path to top of data directory
#         method (str): method to use for normalization. Supported methods
#             are in the normalization_methods database table.

#     Returns:
#         tuple[int, Path, Path]: the resulting dataset id,
#             the path to the normalized data,
#             and the path to the statistics used for normalization
#     """
#     logger.info_(
#         "Beginning dataset normalization at %s using %s...",
#         data_path,
#         method,
#     )

#     assert is_valid_dataset(data_path=data_path, outside_logger=logger)

#     # get stats from training data
#     channel_means, channel_stds = collect_stats(
#         os.path.join(data_path, "train")
#     )

#     # log stats
#     logger.debug_(
#         "channel_means: %s, channel_stds: %s", channel_means, channel_stds
#     )

#     # normalize data in each split
#     for split in ["train", "val", "test"]:
#         for cur_dir, _, files in os.walk(os.path.join(data_path, split)):
#             for file in files:
#                 filepath = os.path.join(cur_dir, file)
#                 im = torch.load(
#                     filepath, map_location="cpu", weights_only=True
#                 )
#                 im = normalize_tensor(
#                     tensor=im,
#                     means=channel_means,
#                     stds=channel_stds,
#                 )
#                 torch.save(im, f=filepath)
#                 logger.debug_("Normalized %s", filepath)

#     # register the newly normalized dataset to the database
#     dataset_id = register_dataset(
#         data_path=data_path, notes="Normalized data"
#     )

#     logger.info_("Done normalizing dataset!")
#     return dataset_id, Path(data_path), Path(data_path)


def minmax_normalize(data_path: Path | str) -> tuple[int, Path, None]:
    """
    Normalizes data in train, val, and test directories at data_path
    by subtracting the minimum and dividing by the new maximum.
    The minimum and maximum are assumed to be 0 and 255
    for all channels,
    and the data is assumed to be in .pt or .npy format.

    Args:
        data_path (Path | str): path to top of data directory

    Returns:
        tuple[int, Path, None]: the resulting dataset id,
            the path to the normalized data,
            and None as a placeholder for the statistics path
    """
    logger.info_(
        "Beginning dataset normalization at %s using min-max method...",
        data_path,
    )

    assert is_valid_dataset(data_path=data_path, outside_logger=logger)

    return 0, Path(data_path), None


def minmaxplus_normalize(data_path: Path | str) -> tuple[int, Path, None]:
    """
    Normalizes data in train, val, and test directories at data_path
    by subtracting the minimum and dividing by the new maximum,
    then shifting and scaling the data to the range [-1, 1]

    Args:
        data_path (Path | str): path to top of data directory

    Returns:
        tuple[int, Path, Path]: the resulting dataset id,
            the path to the normalized data,
            and the path to the statistics used for normalization
    """
    logger.info_(
        "Beginning dataset normalization at %s using min-max-plus method...",
        data_path,
    )

    assert is_valid_dataset(data_path=data_path, outside_logger=logger)

    return 0, Path(data_path), None


def zscore_normalize(data_path: Path | str) -> tuple[int, Path, Path]:
    """
    Normalizes data in train, val, and test directories at data_path
    by subtracting the mean and dividing by the standard deviation
    on a per-channel basis

    Args:
        data_path (Path | str): path to top of data directory

    Returns:
        tuple[int, Path, Path]: the resulting dataset id,
            the path to the normalized data,
            and the path to the statistics used for normalization
    """
    logger.info_(
        "Beginning dataset normalization at %s using z-score method...",
        data_path,
    )

    assert is_valid_dataset(data_path=data_path, outside_logger=logger)

    return 0, Path(data_path), Path(data_path)


def pixelz_normalize(data_path: Path | str) -> tuple[int, Path, Path]:
    """
    Normalizes data in train, val, and test directories at data_path
    by subtracting the mean and dividing by the standard deviation
    on a per-pixel basis

    Args:
        data_path (Path | str): path to top of data directory

    Returns:
        tuple[int, Path, Path]: the resulting dataset id,
            the path to the normalized data,
            and the path to the statistics used for normalization
    """
    logger.info_(
        "Beginning dataset normalization at %s using pixelz method...",
        data_path,
    )

    assert is_valid_dataset(data_path=data_path, outside_logger=logger)

    return 0, Path(data_path), Path(data_path)


def normalize_data(
    data_path: Path | str, method: str
) -> tuple[int, Path, Path | None]:
    """
    Normalizes data in train, val, and test directories at data_path
    according to statistics of the train data using provided method

    Args:
        data_path (Path | str): path to top of data directory
        method (str): method to use for normalization. Supported methods
            are in the normalization_methods database table.

    Returns:
        tuple[int, Path, Path]: the resulting dataset id,
            the path to the normalized data,
            and the path to the statistics used for normalization
    """
    assert is_valid_dataset(data_path=data_path, outside_logger=logger)

    # make sure the method is in the normalization_methods database table
    postgres_connection_string = get_postgres_connection_string()
    # pylint has a false positive when using psycopg 3's new context managers
    # pylint: disable=not-context-manager
    with connect(postgres_connection_string, autocommit=True) as conn:
        with conn.cursor() as curs:
            curs.execute(
                "SELECT method FROM normalization_methods WHERE method=%s;",
                (method,),
            )
            res = curs.fetchone()
            if res is None:
                # if method is not in the database, raise an error
                # and provide the methods currently in the database
                curs.execute("SELECT method FROM normalization_methods;")
                methods = curs.fetchall()
                message = (
                    f"Provided normalization method '{method}' "
                    "is not in the normalization_methods database table. "
                    f"Methods currently in the database are: {methods}."
                )
                logger.error_("%s", message)
                raise ValueError(message)
    # pylint: enable=not-context-manager

    # normalize data using the specified method
    if method == "minmax":
        return minmax_normalize(data_path=data_path)
    if method == "minmaxplus":
        return minmaxplus_normalize(data_path=data_path)
    if method == "zscore":
        return zscore_normalize(data_path=data_path)
    if method == "pixelz":
        return pixelz_normalize(data_path=data_path)

    # if method is not supported, raise an error
    message = (
        f"Provided normalization method '{method}' is not implemented. "
        "Implement it in data_processing.py and add it to if statements above."
    )
    logger.error_("%s", message)
    raise ValueError(message)


def convert_to_jpeg(
    data_path: Path | str, quality: float, out_path: Path | str | None = None
) -> tuple[int, Path]:
    """
    Converts all tensors in data_path to JPEG images
    with the given quality and saves them in out_path

    Args:
        data_path (Path | str): path to the top of the data directory
        quality (float): quality of the JPEG images to save.
            Must be between 0 and 100. See pillow docs for more info.
        out_path (Path | str | None, optional): path where data
            should be saved to. Defaults to data_path + '-c[job_id]'.

    Returns:
        tuple[int, Path]: the dataset ID and the path to the split data
    """
    # specify default out_path if not given
    job_id = get_data_processing_job_id()
    out_path = Path(str(data_path).removesuffix("/") + "-s" + str(job_id))

    logger.info_(
        "Converting data at %s to JPEG at %s using compression quality %s",
        data_path,
        out_path,
        quality,
    )

    assert is_valid_dataset(data_path=data_path, outside_logger=logger)

    # TODO: implement conversion to JPEG

    return job_id, Path(out_path)
