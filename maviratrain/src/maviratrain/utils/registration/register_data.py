"""
Procedures to register a dataset or data processing job
to the PostgreSQL database.
"""

import re
from os import listdir, walk
from os.path import join
from pathlib import Path
from typing import Any

from psycopg import connect

from ..general import (
    get_data_processing_job_id,
    get_loading_function,
    get_logger,
    get_postgres_connection_string,
    get_time,
    is_valid_dataset,
)

# set up logger
logger = get_logger(
    "mt.utils.registration.register_data",
    log_filename="../logs/data_processing/data_processing.log",
    rotation_params=(1000000, 1000),  # 1 MB, 1000 backups
)


def get_dataset_id(data_path: Path | str) -> int:
    """
    Gets the ID of a dataset in the PostgreSQL database.

    Args:
        data_path (str): path to the directory that contains the dataset

    Returns:
        int: the ID of the dataset in the database
    """
    # add trailing slash to data path because it is expected in the database
    data_path = str(data_path).strip("/") + "/"

    logger.info_("Getting dataset ID for dataset at %s...", data_path)

    # connect to database to get dataset ID
    postgres_connection_string = get_postgres_connection_string()
    # pylint has a false positive when using psycopg 3's context managers
    # pylint: disable=not-context-manager
    with connect(postgres_connection_string) as conn:
        with conn.cursor() as curs:
            curs.execute(
                "SELECT id FROM datasets WHERE dir = %s;", (str(data_path),)
            )
            res = curs.fetchone()
            if res is None:
                message = "Could not find dataset in database"
                logger.error_("%s", message)
                raise RuntimeError(message)
            dataset_id = res[0]
    # pylint: enable=not-context-manager

    logger.info_("Dataset ID found successfully!")
    return dataset_id


def count_images(path: Path | str) -> int:
    """
    Counts the number of images in a directory,
    including any subdirectories.

    Args:
        path (str): path to the directory that contains
            images to be counted

    Returns:
        int: the number of images in the directory
            and any subdirectories
    """
    path = Path(path)
    assert is_valid_dataset(data_path=path)  # check dataset directory

    # iterate through all files in the directory and subdirectories
    count = 0
    for _, _, files in walk(path):
        for _ in files:
            count += 1

    return count


def parse_directory(
    path: Path | str,
) -> tuple[int, int, bool, list[int | None], str]:
    """
    Parses a directory to get information about the dataset.

    Args:
        path (str): path to the directory that contains the dataset

    Returns:
        tuple: number of classes, number of images, whether or not the
            dataset is split, the number of images in
            each type of split, and the filename extension of the images
    """
    path = Path(path)
    assert is_valid_dataset(data_path=path)  # check dataset directory

    is_split = False
    # check if the dataset is split into train/val/test
    if (
        "train" in listdir(path)
        and "val" in listdir(path)
        and "test" in listdir(path)
    ):
        is_split = True

    num_train, num_val, num_test = None, None, None  # initialize counts

    # naively count classes (assume no split)
    num_classes = len(listdir(path))

    # count images in each split if dataset is split
    if is_split:
        num_train = count_images(join(path, "train"))
        num_val = count_images(join(path, "val"))
        num_test = count_images(join(path, "test"))
        num_images = num_train + num_val + num_test

        # get list of classes in each split
        train_list = listdir(join(path, "train"))
        val_list = listdir(join(path, "val"))
        test_list = listdir(join(path, "test"))

        # revise class count based on number of classes in train split
        num_classes = len(train_list)

        # check that each split has the same number of classes
        assert num_classes == len(val_list), (
            f"Train directory has {num_classes} classes, but Val directory "
            f"has {len(listdir(join(path, 'val')))} classes."
        )
        assert num_classes == len(test_list), (
            f"Train directory has {num_classes} classes, but Test directory "
            f"has {len(listdir(join(path, 'test')))} classes."
        )

        # get the filename extension of the images
        extensions = set()
        for _, _, files in walk(join(path, "train")):
            for file in files:
                extensions.add(file.split(".")[-1])
        for _, _, files in walk(join(path, "val")):
            for file in files:
                extensions.add(file.split(".")[-1])
        for _, _, files in walk(join(path, "test")):
            for file in files:
                extensions.add(file.split(".")[-1])
    else:
        # if dataset is not split, count classes and images at base path
        class_list = listdir(path)
        num_classes = len(class_list)
        num_images = count_images(path)

        # get the filename extension of the images
        extensions = set()
        for _, _, files in walk(path):
            for file in files:
                extensions.add(file.split(".")[-1])

    # if dataset has multiple extensions, check that they are png and jpg
    # because this is the only case where multiple extensions are allowed
    if not (
        len(extensions) == 2 and "png" in extensions and "jpg" in extensions
    ):
        # check that all images have the same extension
        assert (
            len(extensions) == 1
        ), f"Found multiple extensions in dataset: {extensions}"
    else:
        # if multiple extensions are png and jpg, only keep one
        extensions = {"jpg"}

    return (
        num_classes,
        num_images,
        is_split,
        [num_train, num_val, num_test],
        extensions.pop(),
    )


def register_dataset(
    data_path: Path | str,
    previous_dataset_id: int | None = None,
    job_id: int | None = None,
    notes: str | None = None,
) -> int:
    """
    Registers a dataset to the PostgreSQL database.

    Args:
        data_path (str): path to the directory that contains the dataset
        previous_dataset_id (int, optional): ID of the dataset that was
            used to create the current dataset. Defaults to None.
        job_id (int, optional): ID of the current data processing job.
            Defaults to None.
        notes (str, optional): notes about the dataset.
            Defaults to None.

    Returns:
        int: the ID of the newly registered dataset in the database
    """
    # add trailing slash to data path if not already present for consistency
    data_path = str(data_path).strip("/") + "/"

    logger.info_("Registering dataset at %s in database...", data_path)

    assert is_valid_dataset(data_path=data_path)  # check dataset directory

    # get initial information about the dataset
    num_classes, num_images, is_split, split_sizes, extension = (
        parse_directory(data_path)
    )

    # get loading function based on extension
    load_func = get_loading_function(extension)

    # get previous dataset info if available
    if previous_dataset_id is not None:
        # connect to database to get info for previous dataset and current job
        postgres_connection_string = get_postgres_connection_string()
        # pylint has a false positive when using psycopg 3's context managers
        # pylint: disable=not-context-manager
        with connect(postgres_connection_string) as conn:
            with conn.cursor() as curs:
                # get previous dataset info
                curs.execute(
                    "SELECT ("
                    "is_cleaned, is_resized, is_split, is_normalized, "
                    "is_converted, resize_height, resize_width, norm_method"
                    ") FROM datasets WHERE id = %s;",
                    (previous_dataset_id,),
                )
                # save previous dataset info as tuple
                prev_dataset_info = curs.fetchone()
                if prev_dataset_info is None:
                    message = "Could not find previous dataset in database"
                    logger.error_("%s", message)
                    raise RuntimeError(message)
        # pylint: enable=not-context-manager

    # get job info if available
    if job_id is not None:
        postgres_connection_string = get_postgres_connection_string()
        # pylint has a false positive when using psycopg 3's context managers
        # pylint: disable=not-context-manager
        with connect(postgres_connection_string) as conn:
            with conn.cursor() as curs:
                # get current job info
                curs.execute(
                    "SELECT ("
                    "cleaning, resizing, splitting, "
                    "normalization, conversion, "
                    "resize_height, resize_width, norm_method"
                    ") FROM data_processing_jobs WHERE id = %s;",
                    (job_id,),
                )
                # save job info as tuple
                job_info = curs.fetchone()
                if job_info is None:
                    message = "Could not find job in database"
                    logger.error_("%s", message)
                    raise RuntimeError(message)
        # pylint: enable=not-context-manager

    # parse previous dataset info
    if previous_dataset_id is not None:
        # if previous dataset, unpack variables from tuple
        (
            prev_is_cleaned,
            prev_is_resized,
            prev_is_split,
            prev_is_normalized,
            prev_is_converted,
            prev_image_height,
            prev_image_width,
            prev_norm_method,
        ) = prev_dataset_info  # type: ignore
        # MyPy doesn't recognize that prev_dataset_info cannot be None here
    else:
        # if no previous dataset, initialize variables
        prev_is_cleaned = None
        prev_is_resized = None
        prev_is_split = None
        prev_is_normalized = None
        prev_is_converted = None
        prev_image_height = None
        prev_image_width = None
        prev_norm_method = None

    # parse job info
    if job_id is not None:
        # if job, unpack variables from tuple
        (
            cleaning,
            resizing,
            splitting,
            normalization,
            conversion,
            job_image_height,
            job_image_width,
            job_norm_method,
        ) = job_info[  # type: ignore
            0
        ]
        # MyPy doesn't recognize that job_info cannot be None here
    else:
        # if no job, initialize variables
        cleaning = None
        resizing = None
        splitting = None
        normalization = None
        conversion = None
        job_image_height = None
        job_image_width = None
        job_norm_method = None

    # adjust dataset info based on job info

    if not prev_is_cleaned and cleaning:
        # just cleaned dataset
        is_cleaned = True
    elif prev_is_cleaned is not None:
        # if not cleaning, keep previous value
        is_cleaned = prev_is_cleaned
    else:
        # if no previous value, initialize to None
        is_cleaned = None

    if is_cleaned and not prev_is_resized and resizing:
        # just resized dataset
        is_resized = True
        image_height = job_image_height
        image_width = job_image_width
    elif prev_is_resized is not None:
        # if not resizing, keep previous values
        is_resized = prev_is_resized
        image_height = prev_image_height
        image_width = prev_image_width
    else:
        # if no previous values, initialize to None
        is_resized = None
        image_height = None
        image_width = None

    if is_resized and not prev_is_split and splitting:
        # just split dataset
        is_split = True
    elif prev_is_split is not None:
        # if not splitting, keep previous value
        is_split = prev_is_split
    # no else statement because
    # is_split is already initialized in line 222 during parse_directory call

    if is_split and not prev_is_normalized and normalization:
        # just normalized dataset
        is_normalized = True
        norm_method = job_norm_method
    elif prev_is_normalized is not None:
        # if not normalizing, keep previous values
        is_normalized = prev_is_normalized
        norm_method = prev_norm_method
    else:
        # if no previous values, initialize to None
        is_normalized = None
        norm_method = None

    if is_normalized and not prev_is_converted and conversion:
        # just converted dataset
        is_converted = True
    elif prev_is_converted is not None:
        # if not converting, keep previous value
        is_converted = prev_is_converted
    else:
        # if no previous value, initialize to None
        is_converted = None

    # if we don't know if dataset is cleaned, check filenames for cleaning
    if is_cleaned is None:
        break_flag = False  # flag to break out of nested loop
        # cleaned files are of the form im#####.ext
        clean_format = re.compile(r"^im\d{5}\.[a-zA-Z]{2,4}$")

        # check if all files match the correct format
        for _, _, files in walk(data_path):
            for file in files:
                # if any file isn't cleaned, label dataset as not cleaned
                if re.match(clean_format, file) is None:
                    is_cleaned = False
                    break_flag = True  # set flag to break out of nested loop
                    break
            if break_flag:
                break  # break out of outer loop if inner loop breaks

        # if all files are cleaned, label dataset as cleaned
        if is_cleaned is None:
            is_cleaned = True

    # if we don't know if dataset is resized, open images to check dimensions
    if is_resized is None and not is_cleaned:
        is_resized = False  # if not cleaned, shouldn't be resized
    elif is_resized is None:
        break_flag = False  # flag to break out of nested loop
        # check if all images have the same dimensions
        for cur_dir, _, files in walk(data_path):
            for file in files:
                # load image to get dimensions
                im = load_func(join(cur_dir, file))
                height, width = im.shape[-2:]

                # initialize image dimensions if not already set
                if image_height is None or image_width is None:
                    image_height, image_width = height, width
                # if already initialized dimensions, compare to current image
                elif image_height != height or image_width != width:
                    logger.debug_(
                        "Found images with different dimensions. "
                        f"Found {height}x{width} "
                        f"and {image_height}x{image_width}."
                    )
                    is_resized = False
                    break_flag = True  # set flag to break out of nested loop
                    # set image dimensions to None if not consistent
                    image_height, image_width = None, None
                    break

            if break_flag:
                break  # break out of outer loop if inner loop breaks

    # if we don't know if dataset is normalized, check for normalization
    if is_normalized is None and (
        not is_cleaned or not is_resized or not is_split
    ):
        # shouldn't be normalized if not cleaned, resized, or split
        is_normalized = False
    elif is_normalized is None:
        # initialize is_normalized to True
        is_normalized = True

        # initialize norm_method flags
        z_flag = True
        minmax_flag = True
        minmax_extended_flag = True

        # check training data for normalization
        for cur_dir, _, files in walk(join(data_path, "train")):
            for file in files:
                im = load_func(join(cur_dir, file))

                # check if channel means are not close to 0
                if z_flag and im.mean() > 5:
                    # if not, z-score normalization was not used
                    z_flag = False
                # or if all values are not between 0 and 1
                if minmax_flag and im.min() < 0 and im.max() > 1:
                    # if not, min-max normalization was not used
                    minmax_flag = False
                # or if all values are not between -1 and 1
                if minmax_extended_flag and im.min() < -1 and im.max() > 1:
                    # if not, min-max-plus normalization was not used
                    minmax_extended_flag = False
            if not z_flag and not minmax_flag:
                is_normalized = False
                logger.debug_(
                    "Filters for z-score and min-max normalization failed. "
                    "If the dataset is normalized, please fix the filters."
                )
                break  # break out of loop if both flags are False
        if is_normalized:
            # if here, data must be normalized
            if z_flag:
                norm_method = "z_unknown"
            elif minmax_flag:
                norm_method = "minmax_unknown"
            elif minmax_extended_flag:
                norm_method = "minmaxextended_unknown"

    # if we don't know if dataset is converted, check for conversion
    if is_converted is None and (
        not is_cleaned or not is_resized or not is_split or not is_normalized
    ):
        # shouldn't be converted if not cleaned, resized, split, and normalized
        is_converted = False
    elif is_converted is None and extension in ("pt", "pth"):
        # if extension is pt or pth, dataset is not converted
        is_converted = False
    elif is_converted is None:
        # if extension is not pt or pth, dataset is converted
        is_converted = True

    # insert database record into datasets table
    postgres_connection_string = get_postgres_connection_string()
    # pylint has a false positive when using psycopg 3's new context managers
    # pylint: disable=not-context-manager
    with connect(postgres_connection_string, autocommit=True) as conn:
        with conn.cursor() as curs:
            curs.execute(
                """
                INSERT INTO datasets (
                    dir,
                    extension,
                    created,
                    num_classes,
                    num_images,
                    is_split,
                    is_cleaned,
                    is_resized,
                    is_normalized,
                    is_converted,
                    num_train,
                    num_val,
                    num_test,
                    image_height,
                    image_width,
                    norm_method,
                    notes
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s
                );
                """,
                (
                    data_path,
                    extension,
                    get_time(),
                    num_classes,
                    num_images,
                    is_split,
                    is_cleaned,
                    is_resized,
                    is_normalized,
                    is_converted,
                    split_sizes[0],
                    split_sizes[1],
                    split_sizes[2],
                    image_height,
                    image_width,
                    norm_method,
                    notes,
                ),
            )
            curs.execute("SELECT MAX(id) FROM datasets;")
            result = curs.fetchone()
            # ignore type because psycopg returns (None,) if no datasets
            # but Pylance expects just None
            # this happens because we are using the MAX function
            if result[0] is None:  # type: ignore
                message = "Failed to get dataset ID"
                logger.error_("%s", message)
                raise RuntimeError(message)
            dataset_id = result[0]  # type: ignore
    # pylint: enable=not-context-manager

    logger.info_("Dataset registered successfully!")
    return dataset_id


def register_processing_job(
    result_dict: dict[str, Any],
    notes: str | None = None,
) -> int:
    """
    Registers a data processing job to the PostgreSQL database

    Args:
        result_dict (dict[str, Any]): dictionary containing
            information about the data processing job
        notes (str | None, optional): Any notes to include.
            Defaults to None.
    """
    # get job ID based on the previous job ID
    job_id = get_data_processing_job_id(new=True)

    logger.info_("Registering data processing job %s in database...", job_id)

    # parse result_dict for dataset IDs
    if result_dict["0"] is not None:
        cleaned_dataset_id = result_dict["0"][0]
    else:
        cleaned_dataset_id = None
    if result_dict["1"] is not None:
        resized_dataset_id = result_dict["1"][0]
    else:
        resized_dataset_id = None
    if result_dict["2"] is not None:
        split_dataset_id = result_dict["2"][0]
    else:
        split_dataset_id = None
    if result_dict["3"] is not None:
        normalized_dataset_id = result_dict["3"][0]
    else:
        normalized_dataset_id = None
    if result_dict["4"] is not None:
        converted_dataset_id = result_dict["4"][0]
    else:
        converted_dataset_id = None

    # insert database record into processing_jobs table
    postgres_connection_string = get_postgres_connection_string()
    # pylint has a false positive when using psycopg 3's new context managers
    # pylint: disable=not-context-manager
    with connect(postgres_connection_string, autocommit=True) as conn:
        with conn.cursor() as curs:
            curs.execute(
                """
                INSERT INTO data_processing_jobs (
                    start_time,
                    end_time,
                    cleaning,
                    resizing,
                    splitting,
                    normalization,
                    conversion,
                    starting_dataset_path,
                    cleaned_dataset_id,
                    resized_dataset_id,
                    split_dataset_id,
                    normalized_dataset_id,
                    converted_dataset_id,
                    resize_height,
                    resize_width,
                    interpolation,
                    seed,
                    cleanup,
                    train_percent,
                    val_percent,
                    test_percent,
                    stats_path,
                    norm_method,
                    conversion_format,
                    jpeg_quality,
                    notes
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s
                );
                """,
                (
                    result_dict["start_time"],
                    result_dict["end_time"],
                    0 in result_dict["stages"],
                    1 in result_dict["stages"],
                    2 in result_dict["stages"],
                    3 in result_dict["stages"],
                    4 in result_dict["stages"],
                    result_dict["starting_dataset_path"],
                    cleaned_dataset_id,
                    resized_dataset_id,
                    split_dataset_id,
                    normalized_dataset_id,
                    converted_dataset_id,
                    result_dict["resize_height"],
                    result_dict["resize_width"],
                    result_dict["interpolation"],
                    result_dict["seed"],
                    result_dict["cleanup"],
                    int(result_dict["ratios"][0]),
                    int(result_dict["ratios"][1]),
                    int(result_dict["ratios"][2]),
                    result_dict["stats_path"],
                    result_dict["norm_method"],
                    result_dict["conversion_format"],
                    result_dict["jpeg_quality"],
                    notes,
                ),
            )

    logger.info_("Data processing job registered successfully!")
    return job_id


def update_processing_job(
    job_id: int,
    result_dict: dict[str, Any],
) -> int:
    """
    Updates a data processing job in the PostgreSQL database

    Args:
        job_id (int): ID of the data processing job to update
        result_dict (dict[str, Any]): dictionary containing
            information about the data processing job
        notes (str | None, optional): Any notes to include.
            Defaults to None.
    """
    logger.info_("Updating data processing job %s in database...", job_id)

    # parse result_dict for dataset IDs
    if result_dict["0"] is not None:
        cleaned_dataset_id = result_dict["0"][0]
    else:
        cleaned_dataset_id = None
    if result_dict["1"] is not None:
        resized_dataset_id = result_dict["1"][0]
    else:
        resized_dataset_id = None
    if result_dict["2"] is not None:
        split_dataset_id = result_dict["2"][0]
    else:
        split_dataset_id = None
    if result_dict["3"] is not None:
        normalized_dataset_id = result_dict["3"][0]
    else:
        normalized_dataset_id = None
    if result_dict["4"] is not None:
        converted_dataset_id = result_dict["4"][0]
    else:
        converted_dataset_id = None

    # insert database record into processing_jobs table
    postgres_connection_string = get_postgres_connection_string()
    # pylint has a false positive when using psycopg 3's new context managers
    # pylint: disable=not-context-manager
    with connect(postgres_connection_string, autocommit=True) as conn:
        with conn.cursor() as curs:
            curs.execute(
                """
                UPDATE data_processing_jobs
                SET
                    end_time = %s,
                    cleaned_dataset_id = %s,
                    resized_dataset_id = %s,
                    split_dataset_id = %s,
                    normalized_dataset_id = %s,
                    converted_dataset_id = %s,
                    stats_path = %s
                WHERE id = %s;
                """,
                (
                    result_dict["end_time"],
                    cleaned_dataset_id,
                    resized_dataset_id,
                    split_dataset_id,
                    normalized_dataset_id,
                    converted_dataset_id,
                    str(result_dict["stats_path"]),
                    job_id,
                ),
            )

    logger.info_("Data processing job updated successfully!")
    return job_id
