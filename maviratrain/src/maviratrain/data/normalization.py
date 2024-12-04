"""
Procedures for cleaning and processing data
and generating train/val/test splits.
"""

import os
from pathlib import Path

import torch
from numpy import load as np_load

from ..utils.general import (
    get_data_processing_job_id,
    get_dataset_extension,
    get_logger,
    get_save_function,
    is_valid_dataset,
)

# set up logger
logger = get_logger(
    "mt.data.normalization",
    # should be running from a notebook, hence the ../
    log_filename="../logs/data_processing/normalization.log",
    rotation_params=(1000000, 1000),  # 1 MB, 1000 backups
)


def minmax_normalize(
    data_path: Path | str, out_path: Path | str
) -> tuple[Path, None]:
    """
    Normalizes data in train, val, and test directories at data_path
    by subtracting the minimum and dividing by the new maximum
    and saving the normalized data at out_path.
    The minimum and maximum are assumed to be 0 and 255
    for all channels,
    and the data is assumed to be in .pt or .npy format.

    Args:
        data_path (Path | str): path to top of data directory
        out_path (Path | str): path where data should be saved to.
            If the same as data_path, data will be normalized in place.

    Returns:
        tuple[Path, None]: the path to the normalized data
            and None as a placeholder for the statistics path
    """
    logger.info_(
        "Beginning dataset normalization at %s using min-max method...",
        data_path,
    )

    assert is_valid_dataset(
        data_path=data_path, split_test=True, outside_logger=logger
    )
    assert is_valid_dataset(
        data_path=out_path, split_test=False, outside_logger=logger
    )

    out_path = str(out_path).strip("/") + "/"  # ensure trailing slash

    # get the file extension for saving later
    extension = get_dataset_extension(data_path)
    save_func = get_save_function(extension)

    # loop through all files in data_path
    for cur_dir, sub_dirs, files in os.walk(data_path):
        # create the subdirectory structure in out_path if it doesn't exist
        for sub_dir in sub_dirs:
            new_sub_dir_path = os.path.join(
                out_path,
                cur_dir.strip("/")
                .removeprefix(str(data_path).strip("/"))
                .strip("/"),
                sub_dir,
            )
            os.mkdir(new_sub_dir_path)

        # normalize each file
        for filename in files:
            filepath = os.path.join(cur_dir, filename)
            if extension == "pt":
                im = torch.load(
                    filepath, map_location="cpu", weights_only=True
                )
            elif extension == "npy":
                im = torch.from_numpy(np_load(filepath))
            else:
                message = (
                    f"File {filepath} is not of a supported format. "
                    "Currently only .pt and .npy formats are supported."
                )
                logger.error_("%s", message)
                raise ValueError(message)

            # normalize the image assuming global min and max of 0 and 255
            im = im / 255

            # set the filepath for the normalized image
            new_filepath = Path(
                filepath.replace(str(data_path), str(out_path))
            )
            # save the normalized image
            save_func(im, new_filepath)

            logger.debug_(
                "Normalized %s and saved as %s", filepath, new_filepath
            )

    return Path(out_path), None


def minmax_extended_normalize(
    data_path: Path | str, out_path: Path | str
) -> tuple[Path, None]:
    """
    Normalizes data in train, val, and test directories at data_path
    by subtracting the minimum and dividing by the new maximum,
    then shifting and scaling the data to the range [-1, 1].
    The minimum and maximum are assumed to be 0 and 255
    for all channels,
    and the data is assumed to be in .pt or .npy format.

    Args:
        data_path (Path | str): path to top of data directory
        out_path (Path | str): path where data should be saved to.
            If the same as data_path, data will be normalized in place.

    Returns:
        tuple[Path, None]: the path to the normalized data
            and None as a placeholder for
            the path to the statistics used for normalization
    """
    logger.info_(
        "Beginning dataset normalization at %s "
        "using min-max-extended method...",
        data_path,
    )

    assert is_valid_dataset(
        data_path=data_path, split_test=True, outside_logger=logger
    )
    assert is_valid_dataset(
        data_path=out_path, split_test=False, outside_logger=logger
    )

    out_path = str(out_path).strip("/") + "/"  # ensure trailing slash

    # get the file extension for saving later
    extension = get_dataset_extension(data_path)
    save_func = get_save_function(extension)

    # loop through all files in data_path
    for cur_dir, sub_dirs, files in os.walk(data_path):
        # create the subdirectory structure in out_path if it doesn't exist
        for sub_dir in sub_dirs:
            new_sub_dir_path = os.path.join(
                out_path,
                cur_dir.strip("/")
                .removeprefix(str(data_path).strip("/"))
                .strip("/"),
                sub_dir,
            )
            os.mkdir(new_sub_dir_path)

        # normalize each file
        for filename in files:
            filepath = os.path.join(cur_dir, filename)
            if extension == "pt":
                im = torch.load(
                    filepath, map_location="cpu", weights_only=True
                )
            elif extension == "npy":
                im = torch.from_numpy(np_load(filepath))
            else:
                message = (
                    f"File {filepath} is not of a supported format. "
                    "Currently only .pt and .npy formats are supported."
                )
                logger.error_("%s", message)
                raise ValueError(message)

            # normalize the image assuming global min and max of 0 and 255
            im = im / 255
            # shift and scale the data to the range [-1, 1]
            im = 2 * im - 1

            # set the filepath for the normalized image
            new_filepath = Path(
                filepath.replace(str(data_path), str(out_path))
            )
            # save the normalized image
            save_func(im, new_filepath)

            logger.debug_(
                "Normalized %s and saved as %s", filepath, new_filepath
            )

    return Path(out_path), None


def zscore_normalize(
    data_path: Path | str, out_path: Path | str
) -> tuple[Path, Path]:
    """
    Normalizes data in train, val, and test directories at data_path
    by subtracting the mean and dividing by the standard deviation
    on a per-channel basis. Statistics will be saved at
    (relative to out_path) ../stats/[method]-[job_id].pt

    Args:
        data_path (Path | str): path to top of data directory
        out_path (Path | str): path where data should be saved to.
            If the same as data_path, data will be normalized in place.

    Returns:
        tuple[Path, Path]: the path to the normalized data,
            and the path to the statistics used for normalization
    """
    logger.info_(
        "Beginning dataset normalization at %s using z-score method...",
        data_path,
    )

    assert is_valid_dataset(
        data_path=data_path, split_test=True, outside_logger=logger
    )

    assert is_valid_dataset(
        data_path=out_path, split_test=False, outside_logger=logger
    )

    out_path = str(out_path).strip("/") + "/"  # ensure trailing slash

    # get the file extension for saving later
    extension = get_dataset_extension(data_path)
    save_func = get_save_function(extension)

    job_id = get_data_processing_job_id(new=False)

    # initialize stats for tracking
    num_files = 0
    channel_means = torch.tensor([0.0, 0.0, 0.0])
    channel_stds = torch.tensor([0.0, 0.0, 0.0])

    # get stats from all files in training data
    for cur_dir, _, files in os.walk(os.path.join(data_path, "train")):
        for filename in files:
            filepath = os.path.join(cur_dir, filename)
            if extension == "pt":
                im = torch.load(
                    filepath, map_location="cpu", weights_only=True
                )
            elif extension == "npy":
                im = torch.from_numpy(np_load(filepath))
            else:
                message = (
                    f"File {filepath} is not of a supported format. "
                    "Currently only .pt and .npy formats are supported."
                )
                logger.error_("%s", message)
                raise ValueError(message)

            # update stats
            num_files += 1
            channel_means += im.mean(dim=[1, 2])
            channel_stds += im.std(dim=[1, 2])
            # this implementation calculates the average standard deviation
            # within a channel, not the standard deviation of the means

    channel_means /= num_files
    channel_stds /= num_files

    logger.debug_(
        "channel_means: %s, channel_stds: %s", channel_means, channel_stds
    )

    # save the statistics
    stats_path = Path(
        os.path.join(out_path, "..", "stats", f"zscore-j{job_id}")
    )
    torch.save(channel_means, str(stats_path) + "-means.pt")
    torch.save(channel_stds, str(stats_path) + "-stds.pt")

    # reshape the channel_means and channel_stds to be 3D tensors
    channel_means = channel_means[:, None, None]
    channel_stds = channel_stds[:, None, None]

    # loop through all files in data_path and normalize
    for cur_dir, sub_dirs, files in os.walk(data_path):
        # create the subdirectory structure in out_path if it doesn't exist
        for sub_dir in sub_dirs:
            new_sub_dir_path = os.path.join(
                out_path,
                cur_dir.strip("/")
                .removeprefix(str(data_path).strip("/"))
                .strip("/"),
                sub_dir,
            )
            os.mkdir(new_sub_dir_path)

        # normalize each file
        for filename in files:
            filepath = os.path.join(cur_dir, filename)
            if extension == "pt":
                im = torch.load(
                    filepath, map_location="cpu", weights_only=True
                )
            elif extension == "npy":
                im = torch.from_numpy(np_load(filepath))
            else:
                message = (
                    f"File {filepath} is not of a supported format. "
                    "Currently only .pt and .npy formats are supported."
                )
                logger.error_("%s", message)
                raise ValueError(message)

            # normalize the image using global channel statistics
            im = (im - channel_means) / channel_stds

            # set the filepath for the normalized image
            new_filepath = Path(
                filepath.replace(str(data_path), str(out_path))
            )
            # save the normalized image
            save_func(im, new_filepath)

            logger.debug_(
                "Normalized %s and saved as %s", filepath, new_filepath
            )

    return Path(out_path), stats_path


def pixelz_normalize(
    data_path: Path | str, out_path: Path | str
) -> tuple[Path, Path]:
    """
    Normalizes data in train, val, and test directories at data_path
    by subtracting the mean and dividing by the standard deviation
    on a per-pixel basis. Statistics will be saved at
    (relative to out_path) ../stats/[method]-[job_id].pt

    Args:
        data_path (Path | str): path to top of data directory
        out_path (Path | str): path where data should be saved to.
            If the same as data_path, data will be normalized in place.

    Returns:
        tuple[Path, Path]: the path to the normalized data,
            and the path to the statistics used for normalization
    """
    logger.info_(
        "Beginning dataset normalization at %s using pixelz method...",
        data_path,
    )

    assert is_valid_dataset(
        data_path=data_path, split_test=True, outside_logger=logger
    )

    assert is_valid_dataset(
        data_path=out_path, split_test=False, outside_logger=logger
    )

    # get the file extension for saving later
    extension = get_dataset_extension(data_path)
    save_func = get_save_function(extension)

    job_id = get_data_processing_job_id(new=False)

    # initialize stats for tracking
    num_files = 0
    channel_means = torch.tensor([0.0, 0.0, 0.0])
    channel_stds = torch.tensor([0.0, 0.0, 0.0])

    # get stats from all files in training data
    for cur_dir, _, files in os.walk(os.path.join(data_path, "train")):
        for filename in files:
            filepath = os.path.join(cur_dir, filename)
            if extension == "pt":
                im = torch.load(
                    filepath, map_location="cpu", weights_only=True
                )
            elif extension == "npy":
                im = torch.from_numpy(np_load(filepath))
            else:
                message = (
                    f"File {filepath} is not of a supported format. "
                    "Currently only .pt and .npy formats are supported."
                )
                logger.error_("%s", message)
                raise ValueError(message)

            # update stats
            num_files += 1
            channel_means += im
            im_mean_subtracted = im - im.mean(dim=[1, 2], keepdim=True)
            channel_stds += im_mean_subtracted**2
            # this implementation calculates the average standard deviation
            # of each pixel from its own mean for each channel

    channel_means /= num_files
    channel_stds /= num_files

    logger.debug_(
        "pixel_means: %s, pixel_stds: %s", channel_means, channel_stds
    )

    # save the statistics
    stats_path = Path(
        os.path.join(out_path, "..", "stats", f"zscore-j{job_id}")
    )
    torch.save(channel_means, str(stats_path) + "-means.pt")
    torch.save(channel_stds, str(stats_path) + "-stds.pt")

    # reshape the channel_means and channel_stds to be 3D tensors
    channel_means = channel_means[:, None, None]
    channel_stds = channel_stds[:, None, None]

    # loop through all files in data_path and normalize
    for cur_dir, sub_dirs, files in os.walk(data_path):
        # create the subdirectory structure in out_path if it doesn't exist
        for sub_dir in sub_dirs:
            new_sub_dir_path = os.path.join(
                out_path,
                cur_dir.strip("/")
                .removeprefix(str(data_path).strip("/"))
                .strip("/"),
                sub_dir,
            )
            os.mkdir(new_sub_dir_path)

        # normalize each file
        for filename in files:
            filepath = os.path.join(cur_dir, filename)
            if extension == "pt":
                im = torch.load(
                    filepath, map_location="cpu", weights_only=True
                )
            elif extension == "npy":
                im = torch.from_numpy(np_load(filepath))
            else:
                message = (
                    f"File {filepath} is not of a supported format. "
                    "Currently only .pt and .npy formats are supported."
                )
                logger.error_("%s", message)
                raise ValueError(message)

            # normalize the image using global per-pixel statistics
            im = (im - channel_means) / channel_stds

            # set the filepath for the normalized image
            out_path = str(out_path).strip("/") + "/"  # ensure trailing slash
            new_filepath = Path(
                filepath.replace(str(data_path), str(out_path))
            )
            # save the normalized image
            save_func(im, new_filepath)

    return Path(out_path), stats_path


def local_zscore_normalize(
    data_path: Path | str, out_path: Path | str
) -> tuple[Path, None]:
    """
    Normalizes data in train, val, and test directories at data_path
    by subtracting the mean for each channel and dividing by
    the standard deviation for each channel on a per-image basis.
    The data is assumed to be in .pt or .npy format.

    Args:
        data_path (Path | str): path to top of data directory
        out_path (Path | str): path where data should be saved to.
            If the same as data_path, data will be normalized in place.

    Returns:
        tuple[Path, None]: the path to the normalized data
            and None as a placeholder for
            the path to the statistics used for normalization
    """
    logger.info_(
        "Normalizing data at %s using local z-score method "
        "and saving at %s...",
        data_path,
    )

    assert is_valid_dataset(
        data_path=data_path, split_test=True, outside_logger=logger
    )
    assert is_valid_dataset(
        data_path=out_path, split_test=False, outside_logger=logger
    )

    out_path = str(out_path).strip("/") + "/"  # ensure trailing slash

    # get the file extension for saving later
    extension = get_dataset_extension(data_path)
    save_func = get_save_function(extension)

    # loop through all files in data_path
    for cur_dir, sub_dirs, files in os.walk(data_path):
        # create the subdirectory structure in out_path if it doesn't exist
        for sub_dir in sub_dirs:
            new_sub_dir_path = os.path.join(
                out_path,
                cur_dir.strip("/")
                .removeprefix(str(data_path).strip("/"))
                .strip("/"),
                sub_dir,
            )
            os.mkdir(new_sub_dir_path)

        # normalize each file
        for filename in files:
            filepath = os.path.join(cur_dir, filename)
            if extension == "pt":
                im = torch.load(
                    filepath, map_location="cpu", weights_only=True
                )
            elif extension == "npy":
                im = torch.from_numpy(np_load(filepath))
            else:
                message = (
                    f"File {filepath} is not of a supported format. "
                    "Currently only .pt and .npy formats are supported."
                )
                logger.error_("%s", message)
                raise ValueError(message)

            # normalize the image using local statistics
            im = (im - im.mean(dim=[1, 2], keepdim=True)) / im.std(
                dim=[1, 2], keepdim=True
            )

            # set the filepath for the normalized image
            new_filepath = Path(
                filepath.replace(str(data_path), str(out_path))
            )
            # save the normalized image
            save_func(im, new_filepath)

    return Path(out_path), None


def local_minmax_normalize(
    data_path: Path | str, out_path: Path | str
) -> tuple[Path, None]:
    """
    Normalizes data in train, val, and test directories at data_path
    by subtracting the minimum and dividing by the new maximum
    on a per-channel basis for each image.
    Data is assumed to be in .pt or .npy format.

    Args:
        data_path (Path | str): path to top of data directory
        out_path (Path | str): path where data should be saved to.
            If the same as data_path, data will be normalized in place.

    Returns:
        tuple[Path, None]: the path to the normalized data
            and None as a placeholder for
            the path to the statistics used for normalization
    """
    logger.info_(
        "Beginning dataset normalization at %s using local min-max method...",
        data_path,
    )

    assert is_valid_dataset(
        data_path=data_path, split_test=True, outside_logger=logger
    )
    assert is_valid_dataset(
        data_path=out_path, split_test=False, outside_logger=logger
    )

    out_path = str(out_path).strip("/") + "/"  # ensure trailing slash

    # get the file extension for saving later
    extension = get_dataset_extension(data_path)
    save_func = get_save_function(extension)

    # loop through all files in data_path
    for cur_dir, sub_dirs, files in os.walk(data_path):
        # create the subdirectory structure in out_path if it doesn't exist
        for sub_dir in sub_dirs:
            new_sub_dir_path = os.path.join(
                out_path,
                cur_dir.strip("/")
                .removeprefix(str(data_path).strip("/"))
                .strip("/"),
                sub_dir,
            )
            os.mkdir(new_sub_dir_path)

        # normalize each file
        for filename in files:
            filepath = os.path.join(cur_dir, filename)
            if extension == "pt":
                im = torch.load(
                    filepath, map_location="cpu", weights_only=True
                )
            elif extension == "npy":
                im = torch.from_numpy(np_load(filepath))
            else:
                message = (
                    f"File {filepath} is not of a supported format. "
                    "Currently only .pt and .npy formats are supported."
                )
                logger.error_("%s", message)
                raise ValueError(message)

            # normalize the image using local statistics
            for channel in range(im.shape[0]):
                im[channel, :, :] = (
                    im[channel, :, :] - im[channel, :, :].min()
                ) / (im[channel, :, :].max() - im[channel, :, :].min())

            # set the filepath for the normalized image
            new_filepath = Path(
                filepath.replace(str(data_path), str(out_path))
            )
            # save the normalized image
            save_func(im, new_filepath)

    return Path(out_path), None


def local_minmax_extended_normalize(
    data_path: Path | str, out_path: Path | str
) -> tuple[Path, None]:
    """
    Normalizes data in train, val, and test directories at data_path
    by subtracting the minimum and dividing by the new maximum,
    then shifting and scaling the data to the range [-1, 1]
    on a per-channel basis for each image.
    The data is assumed to be in .pt or .npy format.

    Args:
        data_path (Path | str): path to top of data directory
        out_path (Path | str): path where data should be saved to.
            If the same as data_path, data will be normalized in place.

    Returns:
        tuple[Path, None]: the path to the normalized data
            and None as a placeholder for
            the path to the statistics used for normalization
    """
    logger.info_(
        "Beginning dataset normalization at %s "
        "using local min-max-extended method...",
        data_path,
    )

    assert is_valid_dataset(
        data_path=data_path, split_test=True, outside_logger=logger
    )
    assert is_valid_dataset(
        data_path=out_path, split_test=False, outside_logger=logger
    )

    out_path = str(out_path).strip("/") + "/"  # ensure trailing slash

    # get the file extension for saving later
    extension = get_dataset_extension(data_path)
    save_func = get_save_function(extension)

    # loop through all files in data_path
    for cur_dir, sub_dirs, files in os.walk(data_path):
        # create the subdirectory structure in out_path if it doesn't exist
        for sub_dir in sub_dirs:
            new_sub_dir_path = os.path.join(
                out_path,
                cur_dir.strip("/")
                .removeprefix(str(data_path).strip("/"))
                .strip("/"),
                sub_dir,
            )
            os.mkdir(new_sub_dir_path)

        # normalize each file
        for filename in files:
            filepath = os.path.join(cur_dir, filename)
            if extension == "pt":
                im = torch.load(
                    filepath, map_location="cpu", weights_only=True
                )
            elif extension == "npy":
                im = torch.from_numpy(np_load(filepath))
            else:
                message = (
                    f"File {filepath} is not of a supported format. "
                    "Currently only .pt and .npy formats are supported."
                )
                logger.error_("%s", message)
                raise ValueError(message)

            # normalize the image using local statistics
            for channel in range(im.shape[0]):
                im[channel, :, :] = (
                    im[channel, :, :] - im[channel, :, :].min()
                ) / (im[channel, :, :].max() - im[channel, :, :].min())
            # shift and scale the data to the range [-1, 1]
            im = 2 * im - 1

            # set the filepath for the normalized image
            new_filepath = Path(
                filepath.replace(str(data_path), str(out_path))
            )
            # save the normalized image
            save_func(im, new_filepath)

    return Path(out_path), None
