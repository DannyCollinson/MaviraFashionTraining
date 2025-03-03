"""
Procedures for cleaning and processing data
and generating train/val/test splits.
"""

import os
import random
import re

from collections.abc import Callable
from pathlib import Path
from subprocess import run

import torch
from numpy import asarray, load as np_load
from PIL import Image
from psycopg import connect
from torchvision.io import ImageReadMode, decode_image
from torchvision.transforms.v2 import InterpolationMode, Resize

from .normalization import (
    minmax_extended_normalize,
    minmax_normalize,
    pixelz_normalize,
    zscore_normalize,
    local_minmax_normalize,
    local_minmax_extended_normalize,
    local_zscore_normalize,
)
from ..utils.general import (
    get_data_processing_job_id,
    get_logger,
    get_postgres_connection_string,
    get_save_function,
    is_valid_dataset,
    np_save,
)
from ..utils.registration.register_data import register_dataset

# set up logger
logger = get_logger(
    "mt.data.data_processing",
    log_filename=("../logs/data_processing/data_processing.log"),
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
                        "or merge the directories "
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
        job_id (int | None, optional): the job ID to use for the
            data processing job. If None, a new job ID will be generated.

    Returns:
        Path: the original data_path
    """
    logger.info_("Cleaning data naming at %s...", data_path)

    assert is_valid_dataset(data_path=data_path, outside_logger=logger)

    clean_subdirectories(data_path=data_path)
    clean_filenames(data_path=data_path)

    # register the newly cleaned dataset to the database
    dataset_id = register_dataset(
        data_path=data_path, job_id=job_id, notes="Cleaned up naming"
    )

    logger.info_("Done cleaning data names!")
    return dataset_id, Path(data_path)


def resize_images(
    data_path: Path | str,
    size: tuple[int, int] = (224, 224),
    interpolation: str = "bilinear",
    save_format: str = "npy",
    deduplicate: bool = True,
    out_path: Path | str | None = None,
) -> tuple[int, Path]:
    """
    Resizes images in data_path to size and saves them in out_path.
    Optionally deduplicates images by removing identical images.

    Args:
        data_path (Path | str): path to top of data directory to resize
        size (tuple[int, int], optional): dimensions to resize
            images to. Defaults to (224, 224).
        interpolation (str, optional): torchvision interpolation mode to
            use for resizing images. Defaults to "bilinear".
        format (str, optional): format to save resized images in.
            Supported formats are 'npy' and 'pt'. Defaults to 'npy'.
        deduplicate (bool, optional): whether to remove duplicate images.
            Defaults to True.
        out_path (Path | str | None, optional): path to save resized
            images out to. Defaults to None.

    Raises:
        NotImplementedError: if out_path already exists
        ValueError: if interpolation mode is not supported

    Returns:
        tuple[int, Path]: the dataset ID and the path to the resized data
    """
    # specify default out_path if not given
    job_id = get_data_processing_job_id(new=False)
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

    # get save function based on format
    save_func = get_save_function(save_format)

    # set to store image hashes for deduplication
    if deduplicate:
        image_hashes = set()

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

            # hash for deduplication before resizing
            if deduplicate:
                im_hash = hash(asarray(im).tobytes())

            # convert to float before resizing for accuracy
            im = im.to(dtype=torch.float32)
            im = resize(im)

            # get image subpath by removing data_path from filepath
            file_subpath = filepath.removeprefix(str(data_path) + "/")
            file_subpath = file_subpath.removeprefix(str(data_path))
            # create new filepath as outpath/subpath
            new_filepath = os.path.join(out_path, file_subpath)
            new_filepath = new_filepath.replace(".jpg", f".{save_format}")

            # deduplicate images
            if deduplicate:
                if im_hash in image_hashes:  # type: ignore
                    # skip saving to new dataset if a duplicate
                    logger.debug_(
                        "Duplicate %s found, not adding to resized dataset",
                        filepath,
                    )
                    continue

                # add to hash set
                image_hashes.add(im_hash)  # type: ignore

                # save using the specified format
                save_func(im, new_filepath)

                logger.debug_(
                    "Resized %s and saved as %s", filepath, new_filepath
                )

    # register the newly created dataset to the database
    dataset_id = register_dataset(
        data_path=out_path, job_id=job_id, notes="Resized images"
    )

    logger.info_("Done resizing images!")
    return dataset_id, Path(out_path)


def train_val_test(
    data_path: Path | str,
    ratios: tuple[int, int, int] = (64, 16, 20),
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
            train, 10% val, 20% test. Defaults to (64, 16, 20).
        out_path (Path | str | None, optional): path where data
            should be split into. Defaults to data_path + '-s[job_id]'.
        seed (int, optional): seed for the python built-in random
            generator. Defaults to 42.

    Returns:
        tuple[int, Path]: the dataset ID and the path to the split data
    """
    # specify default out_path if not given
    job_id = get_data_processing_job_id(new=False)
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
                run(["cp", filepath, new_filepath], check=True)
                logger.debug_("Copied %s to %s", filepath, new_filepath)

    # register the newly created dataset to the database
    dataset_id = register_dataset(
        data_path=out_path, job_id=job_id, notes="Split into train/val/test"
    )

    logger.info_("Done creating train/val/test splits!")
    return dataset_id, Path(out_path)


def normalize_data(
    data_path: Path | str, method: str, out_path: Path | str | None = None
) -> tuple[int, int, Path, Path | None]:
    """
    Normalizes data in train, val, and test directories at data_path
    according to statistics of the train data using provided method

    Args:
        data_path (Path | str): path to top of data directory
        method (str): method to use for normalization. Supported methods
            are in the normalization_methods database table.
        out_path (Path | str | None, optional): path where data
            should be saved to.
            If None will be set to data_path + '-n[job_id]'.
            Statistics (if used) will be saved at (relative to out_path)
            ../stats/[method]-[job_id].pt

    Returns:
        tuple[int, int, Path, Path]: the processing job ID,
            the resulting dataset ID,
            the path to the normalized data,
            and the path to the statistics used for normalization
    """
    assert is_valid_dataset(
        data_path=data_path, split_test=True, outside_logger=logger
    )

    # specify default out_path if not given
    job_id = get_data_processing_job_id(new=False)
    if out_path is None or out_path == "None":
        out_path = Path(str(data_path).removesuffix("/") + "-n" + str(job_id))

    # make sure out_path directory is in the "data" directory
    assert "/data/" in str(out_path), (
        "Results directory out_path must be in the 'data' directory. "
        f"Got {out_path}."
    )

    # create the out_path directory if it doesn't exist
    if not os.path.exists(out_path):
        os.mkdir(out_path)
        logger.debug_("Created output directory %s", out_path)

    # make sure the method is in the normalization_methods database table
    methods = []
    postgres_connection_string = get_postgres_connection_string()
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

    logger.info_(
        "Beginning normalization of data at %s using %s and saving at %s...",
        data_path,
        method,
        out_path,
    )

    # define type for normalization function
    normalization_function: Callable[
        [Path | str, Path | str], tuple[Path, Path | None]
    ]

    # set normalization function based on method
    if method == "minmax":
        normalization_function = minmax_normalize
    elif method == "minmaxextended":
        normalization_function = minmax_extended_normalize
    elif method == "localminmax":
        normalization_function = local_minmax_normalize
    elif method == "localminmaxextended":
        normalization_function = local_minmax_extended_normalize
    elif method == "zscore":
        normalization_function = zscore_normalize
    elif method == "pixelz":
        normalization_function = pixelz_normalize
    elif method == "localzscore":
        normalization_function = local_zscore_normalize
    else:
        # if method is not supported, raise an error
        message = (
            f"Provided normalization method '{method}' is not implemented. "
            f"Either check for spelling (supported methods are {methods}) "
            "or implement it in data_processing.py "
            "and add it to the set of if statements above."
        )
        logger.error_("%s", message)
        raise ValueError(message)

    # normalize the data
    res_path, stats_path = normalization_function(
        data_path=data_path, out_path=out_path
    )

    # register the newly normalized dataset to the database
    dataset_id = register_dataset(
        data_path=res_path, job_id=job_id, notes="Normalized data"
    )

    logger.info_("Done normalizing dataset!")

    return job_id, dataset_id, res_path, stats_path


def convert_to_jpeg(
    data_path: Path | str, quality: float, out_path: Path | str | None = None
) -> tuple[int, int, Path]:
    """
    Converts all files in data_path to JPEG images
    with the given quality and saves them in out_path

    Args:
        data_path (Path | str): path to the top of the data directory.
        quality (float): quality of the JPEG images to save.
            Must be between 0 and 100. See pillow docs for more info.
        out_path (Path | str | None, optional): path where data
            should be saved to. Defaults to data_path + '-c[job_id]'.

    Returns:
        tuple[int, int, Path]: the processing job ID,
            the resulting dataset ID, and the path to the converted data
    """
    # specify default out_path if not given
    job_id = get_data_processing_job_id(new=False)
    out_path = Path(str(data_path).removesuffix("/") + "-c" + str(job_id))

    logger.info_(
        "Converting data at %s to JPEG at %s using compression quality %s",
        data_path,
        out_path,
        quality,
    )

    assert is_valid_dataset(
        data_path=data_path, split_test=True, outside_logger=logger
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

    for cur_dir, sub_dirs, files in os.walk(data_path):
        # replicate the data_path subdirectory structure
        for sub_dir in sub_dirs:
            # get subdirectory subpath by removing data_path from subdir
            subdir_subpath = cur_dir.removeprefix(str(data_path) + "/")
            subdir_subpath = subdir_subpath.removeprefix(str(data_path))
            # create new subdirectory path as outpath/subpath
            new_subdir_subpath = os.path.join(out_path, subdir_subpath)
            destination_subdir = os.path.join(new_subdir_subpath, sub_dir)
            os.mkdir(destination_subdir)
            logger.debug_("Created destination %s", destination_subdir)

        # convert each file and place in out_path/sub_dir/fname.jpg
        for filename in files:
            filepath = os.path.join(cur_dir, filename)
            if filepath.endswith(".pt"):
                im = torch.load(
                    filepath, map_location="cpu", weights_only=True
                )
            elif filepath.endswith(".npy"):
                im = torch.from_numpy(np_load(filepath))
            else:
                message = (
                    f"File {filepath} is not of a supported format. "
                    "Currently only .pt and .npy formats are supported."
                )
                logger.error_("%s", message)
                raise ValueError(message)
            # convert to uint8 before saving as JPEG
            im = im.to(dtype=torch.uint8)
            # get image subpath by removing data_path from filepath
            file_subpath = filepath.removeprefix(str(data_path) + "/")
            file_subpath = file_subpath.removeprefix(str(data_path))
            # create new filepath as outpath/subpath
            new_filepath = os.path.join(out_path, file_subpath)
            new_filepath = new_filepath.replace(".pt", ".jpg")
            im = im.permute(1, 2, 0).numpy()
            Image.fromarray(im).save(
                new_filepath, optimize=True, keep_rgb=True, quality=quality
            )
            logger.debug_(
                "Converted %s and saved as %s", filepath, new_filepath
            )

    logger.info_("Done converting data to JPEG!")

    # register the newly created dataset to the database
    dataset_id = register_dataset(
        data_path=out_path, job_id=job_id, notes="Converted to JPEG"
    )

    return job_id, dataset_id, Path(out_path)


def convert_to_numpy(
    data_path: Path | str, out_path: Path | str | None = None
) -> tuple[int, int, Path]:
    """
    Converts all files in data_path to NumPy arrays
    and saves them in out_path

    Args:
        data_path (Path | str): path to the top of the data directory.
            Data is assumed to be in .pt format.
        out_path (Path | str | None, optional): path where data
            should be saved to. Defaults to data_path + '-c[job_id]'.

    Returns:
        tuple[int, int, Path]: the processing job ID,
            the resulting dataset ID, and the path to the converted data
    """
    # specify default out_path if not given
    job_id = get_data_processing_job_id(new=False)
    out_path = Path(str(data_path).removesuffix("/") + "-c" + str(job_id))

    logger.info_(
        "Converting data at %s to NumPy array at %s", data_path, out_path
    )

    assert is_valid_dataset(
        data_path=data_path, split_test=True, outside_logger=logger
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

    for cur_dir, sub_dirs, files in os.walk(data_path):
        # replicate the data_path subdirectory structure
        for sub_dir in sub_dirs:
            # get subdirectory subpath by removing data_path from subdir
            subdir_subpath = cur_dir.removeprefix(str(data_path) + "/")
            subdir_subpath = subdir_subpath.removeprefix(str(data_path))
            # create new subdirectory path as outpath/subpath
            new_subdir_subpath = os.path.join(out_path, subdir_subpath)
            destination_subdir = os.path.join(new_subdir_subpath, sub_dir)
            os.mkdir(destination_subdir)
            logger.debug_("Created destination %s", destination_subdir)

        # convert each file and place in out_path/sub_dir/fname.jpg
        for filename in files:
            filepath = os.path.join(cur_dir, filename)
            if filepath.endswith(".pt"):
                im = torch.load(
                    filepath, map_location="cpu", weights_only=True
                )
            elif filepath.endswith(".npy"):
                continue  # already in NumPy format
            else:
                message = (
                    f"File {filepath} is not of a supported format. "
                    "Currently only .pt and .npy formats are supported."
                )
                logger.error_("%s", message)
                raise ValueError(message)
            # get image subpath by removing data_path from filepath
            file_subpath = filepath.removeprefix(str(data_path) + "/")
            file_subpath = file_subpath.removeprefix(str(data_path))
            # create new filepath as outpath/subpath
            new_filepath = os.path.join(out_path, file_subpath)
            new_filepath = new_filepath.replace(".pt", ".npy")
            np_save(im.numpy(), new_filepath)
            logger.debug_(
                "Converted %s and saved as %s", filepath, new_filepath
            )

    logger.info_("Done converting data to NumPy arrays!")

    # register the newly created dataset to the database
    dataset_id = register_dataset(
        data_path=out_path, job_id=job_id, notes="Converted to NumPy array"
    )

    return job_id, dataset_id, Path(out_path)


def convert_to_pytorch(
    data_path: Path | str, out_path: Path | str | None = None
) -> tuple[int, int, Path]:
    """
    Converts all files in data_path to PyTorch tensors
    and saves them in out_path

    Args:
        data_path (Path | str): path to the top of the data directory.
            Data is assumed to be in .npy format.
        out_path (Path | str | None, optional): path where data
            should be saved to. Defaults to data_path + '-c[job_id]'.

    Returns:
        tuple[int, int, Path]: the processing job ID,
            the resulting dataset ID, and the path to the converted data
    """
    # specify default out_path if not given
    job_id = get_data_processing_job_id(new=False)
    out_path = Path(str(data_path).removesuffix("/") + "-c" + str(job_id))

    logger.info_(
        "Converting data at %s to PyTorch tensor at %s", data_path, out_path
    )

    assert is_valid_dataset(
        data_path=data_path, split_test=True, outside_logger=logger
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

    for cur_dir, sub_dirs, files in os.walk(data_path):
        # replicate the data_path subdirectory structure
        for sub_dir in sub_dirs:
            # get subdirectory subpath by removing data_path from subdir
            subdir_subpath = cur_dir.removeprefix(str(data_path) + "/")
            subdir_subpath = subdir_subpath.removeprefix(str(data_path))
            # create new subdirectory path as outpath/subpath
            new_subdir_subpath = os.path.join(out_path, subdir_subpath)
            destination_subdir = os.path.join(new_subdir_subpath, sub_dir)
            os.mkdir(destination_subdir)
            logger.debug_("Created destination %s", destination_subdir)

        # convert each file and place in out_path/sub_dir/fname.jpg
        for filename in files:
            filepath = os.path.join(cur_dir, filename)
            if filepath.endswith(".pt"):
                continue  # already in PyTorch format
            if filepath.endswith(".npy"):
                im = torch.from_numpy(np_load(filepath))
            else:
                message = (
                    f"File {filepath} is not of a supported format. "
                    "Currently only .pt and .npy formats are supported."
                )
                logger.error_("%s", message)
                raise ValueError(message)
            # get image subpath by removing data_path from filepath
            file_subpath = filepath.removeprefix(str(data_path) + "/")
            file_subpath = file_subpath.removeprefix(str(data_path))
            # create new filepath as outpath/subpath
            new_filepath = os.path.join(out_path, file_subpath)
            new_filepath = new_filepath.replace(".npy", ".pt")
            torch.save(im, new_filepath)
            logger.debug_(
                "Converted %s and saved as %s", filepath, new_filepath
            )

    logger.info_("Done converting data to PyTorch tensors!")

    # register the newly created dataset to the database
    dataset_id = register_dataset(
        data_path=out_path, job_id=job_id, notes="Converted to PyTorch tensor"
    )

    return job_id, dataset_id, Path(out_path)
