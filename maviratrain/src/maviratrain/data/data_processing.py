"""
Procedures for cleaning and processing data
and generating train/val/test splits
"""

import os
import random
from pathlib import Path
from typing import Union

import torch
from PIL import Image
from torch import Tensor
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms.v2 import InterpolationMode, Resize

from ..utils.general import (
    get_file_date,
    get_log_time,
    is_valid_directory,
)


def clean_subdirectories(data_path: Union[Path, str]) -> Path:
    """
    Renames subdirectories under data_path to remove '_files' suffix,
    making sure this would not cause duplicates

    Arguments:
        data_path {Union[Path, str]} -- path to top of data directory

    Raises:
        FileExistsError: raised if removing '_files' suffix would
            create duplicate subdirectory names

    Returns:
        Path -- original data_path
    """
    assert is_valid_directory(data_path=data_path)  # check directory exists

    with open(
        "logs/data_processing/subdir_cleaning-" f"{get_log_time()}.log",
        "w",
        encoding="utf-8",
    ) as log:
        # log metadata
        log.write(f"data_path: {data_path}\n")

        for cur_dir, sub_dirs, _ in os.walk(data_path):
            for sub_dir in sub_dirs:
                if "_files" in sub_dir:
                    # check for duplicate creation
                    if sub_dir.removesuffix("_files") in sub_dirs:
                        message = (
                            f"Renaming {sub_dir} would result "
                            "in two directories named "
                            f"{sub_dir.removesuffix('_files')}. "
                            "This is not currently handled."
                        )
                        log.write(message + "\n")
                        raise FileExistsError(message)

                    subdir_path = os.path.join(cur_dir, sub_dir)
                    os.rename(subdir_path, subdir_path.removesuffix("_files"))
                    log.write(
                        f"{subdir_path} is now "
                        f"{subdir_path.removesuffix('_files')}\n"
                    )
        log.write("Done cleaning subdirectory names!\n")
    print("Done cleaning subdirectory names!")
    return Path(data_path)


def clean_filenames(data_path: Union[Path, str]) -> Path:
    """
    Renames raw data files to have consistent formatting: im#####.jpg

    Arguments:
        data_path {Union[Path, str]} -- path to data to rename

    Returns:
        Path -- original data_path
    """
    assert is_valid_directory(data_path=data_path)  # check directory exists

    with open(
        "logs/data_processing/filename_cleaning-" f"{get_log_time()}.log",
        "w",
        encoding="utf-8",
    ) as log:
        # log metadata
        log.write(f"data_path: {data_path}\n")

        for cur_dir, _, files in os.walk(data_path):
            for index, filename in enumerate(files):
                if "images" not in filename or (
                    "jpg" not in filename and "png" not in filename
                ):
                    message = (
                        f"Filename {filename} was not expected. "
                        "Was expecting filename of the form "
                        "images.jpg, images.png, images_###.jpg, "
                        "images_###.png, images_####.jpg, or images_####.png"
                    )
                    log.write(message + "\n")
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
                    log.write(f"{filename} saved as {new_filename}\n")
                    os.remove(os.path.join(cur_dir, filename))
                    log.write(f"{filename} removed\n")
                # otherwise rename file with new filename
                else:
                    old_filepath = os.path.join(cur_dir, filename)
                    new_filepath = os.path.join(cur_dir, new_filename)
                    os.rename(old_filepath, new_filepath)
                    log.write(f"{old_filepath} renamed as {new_filepath}\n")
        log.write("Done cleaning filenames!\n")
    print("Done cleaning filenames!")
    return Path(data_path)


def clean_data_naming(data_path: Union[Path, str]) -> Path:
    """
    Cleans the subdirectory names (remove '_files' suffix)
    and filenames (format as 'im####.jpg)

    Arguments:
        data_path {Union[Path, str]} -- path to data directory

    Returns:
        Path -- original data_path
    """
    assert is_valid_directory(data_path=data_path)  # check directory exists

    with open(
        "logs/data_processing/data_name_cleaning-" f"{get_log_time()}.log",
        "w",
        encoding="utf-8",
    ) as log:
        log.write(f"data_path: {data_path}\n")  # log metadata
        log.write("Cleaning subdirectory names...\n")
        clean_subdirectories(data_path=data_path)
        log.writelines(
            ["Done cleaning subdirectory names!\n", "Cleaning filenames...\n"]
        )
        clean_filenames(data_path=data_path)
        log.writelines(["Done cleaning filenames!\n", "Done cleaning!\n"])
    print("Done cleaning!")
    return Path(data_path)


def resize_images(
    data_path: Union[Path, str],
    size: tuple[int, int] = (224, 224),
    interpolation: str = "bilinear",
    out_path: Union[Path, str, None] = None,
) -> Path:
    """
    Generates resized versions of the images at data_path at the location
    out_path of dimensions 'size'. If out_path is not specified, defaults
    to data_path + '-r[job_id]'.

    Args:
        data_path (Union[Path, str]): path to the original data
        size (tuple[int, int], optional): (height, width) in pixels of resized
            data. Defaults to (224, 224).
        interpolation (str, optional): Torchvision InterpolationMode to use
            for resizing images. Case insensitive. Defaults to 'bilinear'.
        out_path (Union[Path, str, None], optional): path to place
            resized data. Defaults to 'data_path-r[job_id]' if None.

    Returns:
        Path: the output path out_path
    """
    assert is_valid_directory(data_path=data_path)  # check directory exists

    # TODO
    # specify default out_path if not given
    if out_path is None:
        out_path = Path(
            str(data_path).removesuffix("/")
            + "-resized_"
            + str(size[0])
            + "x"
            + str(size[1])
            + "-"
            + get_file_date()
        )

    # TODO
    # make sure destination doesn't already exist to avoid issues
    # see below for how we may handle this in the future
    if os.path.exists(out_path):
        raise NotImplementedError(
            f"Destination subdirectory {out_path} "
            "already exists, which is currently not handled"
        )
    os.mkdir(out_path)

    # get the correct interpolation mode
    interpolation = interpolation.lower()  # for case insensitivity
    interpolation_mode = InterpolationMode.BILINEAR
    if interpolation == "bilinear":
        interpolation_mode = InterpolationMode.BILINEAR
    elif interpolation == "bicubic":
        interpolation_mode = InterpolationMode.BICUBIC
    elif interpolation == "nearest":
        interpolation_mode = InterpolationMode.NEAREST
    elif interpolation == "nearest_exact":
        interpolation_mode = InterpolationMode.NEAREST_EXACT
    else:
        raise ValueError(
            f"Provided interpolation mode '{interpolation}' is not supported. "
            "Supported Torchvision InterpolationModes include BILINEAR, "
            "BICUBIC, NEAREST, and NEAREST_EXACT."
        )

    # define transformation
    resize = Resize(
        size=size, interpolation=interpolation_mode, antialias=True
    )

    with open(
        "logs/data_processing/image_resizing-" f"{get_log_time()}.log",
        "w",
        encoding="utf-8",
    ) as log:
        # log metadata
        log.write(f"data_path: {data_path}\n")
        log.write(f"size: {size}\n")
        log.write(f"out_path: {out_path}\n")
        # log previous actions
        log.write(f"Created output directory {out_path}\n")

        for cur_dir, sub_dirs, files in os.walk(data_path):
            # replicate the data_path subdirectory structure
            for sub_dir in sub_dirs:
                destination_subdir = os.path.join(out_path, sub_dir)
                # TODO
                # make sure destination doesn't already exist to avoid issues
                if os.path.exists(destination_subdir):
                    raise NotImplementedError(
                        f"Destination subdirectory {destination_subdir} "
                        "already exists, which is currently not handled"
                    )
                os.mkdir(destination_subdir)
                log.write(f"Created destination {destination_subdir}\n")
                # can possibly do it more like this in the future
                # try:
                #     os.mkdir(destination_subdir)
                #     log.write(f"Created destination {destination_subdir}\n")
                # except FileExistsError:
                #     log.write(
                #         f"Destination {destination_subdir} "
                #         "already exists, reusing it\n"
                #     )
            # resize each file and place in out_path/sub_dir/fname.pt
            for filename in files:
                filepath = os.path.join(cur_dir, filename)
                im = read_image(filepath, mode=ImageReadMode.RGB)
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
                log.write(f"Resized {filepath} and saved as {new_filepath}\n")
        log.write("Done resizing images!\n")
    print("Done resizing images!")
    return Path(out_path)


def train_val_test(
    data_path: Union[Path, str],
    ratios: tuple[int, int, int] = (60, 15, 25),
    out_path: Union[Path, str, None] = None,
    seed: int = 42,
) -> Path:
    """
    Randomly places files in data_path into train, val, and test files
    at the out_path according to the ratios in 'ratios'.

    Args:
        data_path (Union[Path, str]): path to the
            unsplit and unnormalized data
        ratios (list[int, int, int], optional): percentages of each class to
            split into train/val/test. Each must be >= 0 and <= 100 and must
            have sum=100, e.g., (70, 10, 20) for 70% train, 10% val, 20% test.
            Defaults to (60, 15, 25).
        out_path (Union[Path, str, None], optional): path where data
            should be split into. Defaults to data_path + '-s[job_id]'.
        seed (int, optional): seed for the python built-in random generator.
            Defaults to 42.

    Returns:
        Path: the output path out_path
    """
    assert is_valid_directory(data_path=data_path)  # check directory exists

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

    # TODO
    # specify default out_path if not given
    if out_path is None:
        out_path = Path(
            str(data_path).removesuffix("/") + "-split-" + get_file_date()
        )

    # TODO
    # make sure destination doesn't already exist to avoid issues
    # we may handle this differently in the future, see above
    if os.path.exists(out_path):
        raise NotImplementedError(
            f"Destination subdirectory {out_path} "
            "already exists, which is currently not handled"
        )
    os.mkdir(out_path)

    random.seed(seed)  # set random seed for reproducibility

    with open(
        "logs/data_processing/train_val_test-" f"{get_log_time()}.log",
        "w",
        encoding="utf-8",
    ) as log:
        # log metadata
        log.write(f"data_path: {data_path}\n")
        log.write(f"ratios: {ratios}\n")
        log.write(f"out_path: {out_path}\n")
        log.write(f"seed: {seed}\n")
        # log previous actions
        log.write(f"Created output directory {out_path}\n")
        # create directories for splits
        splits = ["train", "val", "test"]
        for split in splits:
            split_dir = os.path.join(out_path, split)
            os.mkdir(split_dir)
            log.write(f"Created output directory {split_dir}\n")

        for cur_dir, sub_dirs, files in os.walk(data_path):
            # replicate data_path subdirectory format in destination
            for sub_dir in sub_dirs:
                for split in splits:
                    new_dir = os.path.join(out_path, split, sub_dir)
                    os.mkdir(new_dir)
                    log.write(f"Created output directory {new_dir}\n")
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
                log.write(f"Copying {split_nums[i]} files to {split_dir}\n")
                for _ in range(split_nums[i]):
                    # draw total of len(files) files from end of file list
                    filename = shuffled_files.pop()
                    # get current and new filepaths
                    filepath = os.path.join(cur_dir, filename)
                    new_filepath = os.path.join(split_dir, filename)
                    # copy to new filepath
                    os.system(f"cp {filepath} {new_filepath}")
                    log.write(f"Copied {filepath} to {new_filepath}\n")
        log.write("Done creating train/val/test splits!")
    print("Done creating train/val/test splits!")
    return Path(out_path)


def collect_stats(
    data_path: Union[Path, str]
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Collects mins, maxes, means, and std. deviations for each channel

    Arguments:
        data_path {Union[Path, str]} -- path to training data

    Returns:
        tuple[list[float]] -- [channel_mins(3), channel_maxes(3),
            channel_means(3), channel stds(3)]
    """
    assert is_valid_directory(data_path=data_path)  # check directory exists

    # initialize stats for tracking
    num_files = 0
    channel_mins = torch.tensor([255.0, 255.0, 255.0])
    channel_maxes = torch.tensor([0.0, 0.0, 0.0])
    channel_means = torch.tensor([0.0, 0.0, 0.0])
    channel_stds = torch.tensor([0.0, 0.0, 0.0])

    # collect stats from training files
    for cur_dir, _, files in os.walk(os.path.join(data_path)):
        for file in files:
            filepath = os.path.join(cur_dir, file)
            im = torch.load(filepath, map_location="cpu", weights_only=True)
            num_files += 1
            channel_means += im.mean(dim=[1, 2])
            channel_stds += im.std(dim=[1, 2])
            im_mins = torch.amin(im, dim=(1, 2))
            im_maxes = torch.amax(im, dim=(1, 2))
            for channel in range(im.shape[0]):
                if im_mins[channel] < channel_mins[channel]:
                    channel_mins[channel] = im_mins[channel].item()
                if im_maxes[channel] > channel_maxes[channel]:
                    channel_maxes[channel] = im_maxes[channel].item()
    channel_means /= num_files
    channel_stds /= num_files

    return channel_mins, channel_maxes, channel_means, channel_stds


def normalize_tensor(
    tensor: Tensor,
    mins: Tensor,
    maxes: Tensor,
    means: Tensor,
    stds: Tensor,
) -> Tensor:
    """
    Normalizes tensor by applying the formula
    normed_tensor = (tensor / spans - means / spans) / (stds / spans)
                  = (tensor - means) / stds
        where spans = maxes - mins

    Arguments:
        tensor {Tensor} -- tensor to be normalized
        mins {Tensor} -- the minimum values in each channel
        maxes {Tensor} -- _the maximum values in each channel
        means {Tensor} -- the mean values in each channel
        stds {Tensor} -- the std. deviations of each channel's values

    Returns:
        Tensor -- the normalized tensor
    """
    spans = maxes - mins
    spans = spans[:, None, None]
    means = means[:, None, None]
    stds = stds[:, None, None]
    return ((tensor / spans) - (means / spans)) / (stds / spans)


def normalize_data(data_path: Union[Path, str]) -> Path:
    """
    Normalizes data in train, val, and test directories at data_path
    according to statistics of the train data. Scales each channel to 0-1
    then subtracts mean and divides by standard deviation

    Arguments:
        data_path {Union[Path, str]} -- path to data to be normalized

    Returns:
        Path -- the original data_path
    """
    assert is_valid_directory(data_path=data_path)  # check directory exists

    # get stats from training data
    channel_mins, channel_maxes, channel_means, channel_stds = collect_stats(
        os.path.join(data_path, "train")
    )

    with open(
        "logs/data_processing/normalize_data-" f"{get_log_time()}.log",
        "w",
        encoding="utf-8",
    ) as log:
        # log stats
        log.write("Got training data statistics\n")
        log.write(f"channel_mins: {channel_mins}\n")
        log.write(f"channel_maxes: {channel_maxes}\n")
        log.write(f"channel_means: {channel_means}\n")
        log.write(f"channel_stds: {channel_stds}\n")

        # normalize data in each split
        for split in ["train", "val", "test"]:
            for cur_dir, _, files in os.walk(os.path.join(data_path, split)):
                for file in files:
                    filepath = os.path.join(cur_dir, file)
                    im = torch.load(
                        filepath, map_location="cpu", weights_only=True
                    )
                    im = normalize_tensor(
                        tensor=im,
                        mins=channel_mins,
                        maxes=channel_maxes,
                        means=channel_means,
                        stds=channel_stds,
                    )
                    torch.save(im, f=filepath)
                    log.write(f"Normalized tensor at {filepath}\n")
        log.write("Done normalizing data!\n")
    print("Done normalizing data!")
    return Path(data_path)


def run_full_processing_pipeline(
    data_path: Union[Path, str]
) -> dict[str, tuple[Path, str]]:
    """
    Runs subdirectory name cleaning, filename cleaning, resizing,
    train_val_test, and normalization all together

    Arguments:
        data_path {Union[Path, str]} -- path to data to start with

    Returns:
        dict[str, tuple[Path, str]] -- dictionary with info
            about intermediate processing steps
    """
    assert is_valid_directory(data_path=data_path)  # check directory exists

    with open(
        "logs/data_processing/"
        f"run_full_processing_pipeline-{get_log_time()}.log",
        "w",
        encoding="utf-8",
    ) as log:
        log.write("Cleaning data naming...\n")
        original_path = clean_data_naming(data_path=data_path)  # run cleaning
        log.writelines(["Done cleaning data names!\n", "Resizing images...\n"])
        resize_path = resize_images(data_path=original_path)  # run resizing
        log.writelines(
            [
                f"Done resizing images!\nResized images at {resize_path}",
                "Creating training splits...\n",
            ]
        )
        split_path = train_val_test(data_path=resize_path)  # run splitting
        log.writelines(
            [
                f"Done creating training splits!\nSplits at {split_path}",
                "Normalizing data...\n",
            ]
        )
        final_path = normalize_data(data_path=split_path)  # run normalization
        log.writelines(
            [
                f"Done normalizing data!\nNormalized data at {final_path}\n",
                "Done data processing!\n",
            ]
        )
    print("Done data processing!")
    result_dict = {
        "Input": (Path(data_path), "no change"),
        "Cleaning": (Path(original_path), "in place"),
        "Resizing": (Path(resize_path), "new path"),
        "Splitting": (Path(split_path), "new path"),
        "Normalization": (Path(final_path), "in place"),
        "Output": (Path(final_path), "no change"),
    }
    return result_dict
