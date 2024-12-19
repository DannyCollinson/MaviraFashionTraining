""" Common utility functions. """

import datetime
import os

# TODO: remove the pylint disable once the issue is resolved
from collections.abc import Callable  # pylint: disable=import-error
from pathlib import Path

from dotenv import load_dotenv
from numpy import ndarray, save as numpy_save
from psycopg import connect
from torch.backends.mps import is_available as is_mps_available
from torch.cuda import is_available as is_cuda_available

from .constants import EXTENSION_TO_SAVE_FUNC, LOAD_FUNCS, SAVE_FUNCS
from .mavira_logging import MaviraTrainLogger


def get_logger(
    name: str,
    level: str = "debug",
    log_filename: Path | str | None = None,
    msg_formatter: str | None = None,
    rotation_params: tuple[int, int] | None = None,
    both: bool = True,
    console_level: str = "info",
) -> MaviraTrainLogger:
    """
    Returns a logger object with the specified parameters.

    Args:
        name (str): the name of the logger
        level (str, optional): the logging level. Defaults to DEBUG.
            If both is true, this will be the level for the file logger.
        log_filename (Path | str, optional): the path to the log file.
            Defaults to None, which means the logger outputs
            to stderror.
        msg_format (str, optional): the format of the log messages.
            Defaults to
            YYYY-MM-DDTHH:MM:SS+00+00 - name - LEVEL - msg.
            Note that time is in UTC regardless of specified format.
        rotation_params: tuple[int, int] -- the rotation parameters for
            the log file. Defaults to None, which means no rotation.
            The first integer is the maximum size of the log file in
            bytes before rotation, and the second integer is the
            number of backup log files. Will only be used
            if log_filename is specified.
        both (bool, optional): whether to log to both file and console.
            Defaults to True.
        console_level (str, optional): the logging level for
            the console logger if both is true. Defaults to INFO.

    Returns:
        logging.Logger: the specified logger object
    """
    return MaviraTrainLogger(
        name,
        level,
        log_filename,
        msg_formatter,
        rotation_params,
        both,
        console_level,
    )


# set up logger
logger = get_logger("mt.utils.general")


def get_time(timespec: str = "seconds") -> str:
    """
    Returns the current UTC datetime in ISO format truncated to timespec

    Args:
        timespec (str, optional): the level of precision to include
            in the time. Defaults to "seconds".

    Returns:
        str -- the current datetime
    """
    assert timespec in [
        "hours",
        "minutes",
        "seconds",
        "milliseconds",
        "microseconds",
    ], (
        "Provided timespec must be one of 'hours', 'minutes', 'seconds', "
        "'milliseconds', or 'microseconds'."
    )

    return datetime.datetime.now(tz=datetime.timezone.utc).isoformat(
        timespec=timespec
    )


def get_date() -> str:
    """
    Returns the current UTC date in ISO format

    Returns:
        str -- the current date
    """
    return datetime.datetime.now(tz=datetime.timezone.utc).isoformat(
        timespec="seconds"
    )[:10]


def is_valid_directory(
    data_path: Path | str, outside_logger: MaviraTrainLogger | None = None
) -> bool:
    """
    Verifies that data_path points to a directory

    Arguments:
        data_path (Path | str) -- path to check for directory
        logger (MaviraTrainLogger, optional) -- logger for reporting errors

    Returns:
        bool -- True if directory exists, False otherwise
    """
    # make sure data directory exists
    res = os.path.isdir(data_path)

    # log error if logger is provided
    if not res and outside_logger:
        outside_logger.error_(
            "Expected data_path to point to a directory. Got %s.", data_path
        )
    assert res, f"Expected data_path to point to a directory. Got {data_path}."
    return res


def is_valid_dataset(
    data_path: Path | str,
    split_test: bool = False,
    outside_logger: MaviraTrainLogger | None = None,
) -> bool:
    """
    Verifies that data_path points to a subdirectory of the
    "data" directory. Optionally verifies that
    there are train, val, and test subdirectories.

    Args:
        data_path (Path | str): path to check
        logger (MaviraTrainLogger, optional): logger for reporting errors.
            Defaults to None.

    Returns:
        bool: True if the path is a valid dataset directory,
            False otherwise.
    """
    # make sure data_path directory exists
    assert is_valid_directory(
        data_path=data_path, outside_logger=outside_logger
    )

    # make sure data_path directory is in the "data" directory
    res = "/data/" in str(data_path)

    # log error if logger is provided
    if not res and outside_logger:
        outside_logger.error_(
            "Expected data_path to point to a subdirectory of the "
            '"data" directory. Got %s.',
            data_path,
        )

    assert res, (
        "Expected data_path to point to a subdirectory "
        f'of the "data" directory. Got {data_path}.'
    )

    # if split_test is True, make sure there are train, val, and test
    if split_test:
        res = (
            os.path.isdir(os.path.join(data_path, "train"))
            and os.path.isdir(os.path.join(data_path, "val"))
            and os.path.isdir(os.path.join(data_path, "test"))
        )

        # log error if logger is provided
        if not res and outside_logger:
            outside_logger.error_(
                "Expected data_path to have "
                "subdirectories 'train', 'val', and 'test'."
            )

        assert res, (
            "Expected data_path to have "
            "subdirectories 'train', 'val', and 'test'."
        )

    return res


def get_dataset_extension(data_path: Path | str) -> str:
    """
    Returns the extension of the files in the dataset directory

    Args:
        data_path (Path | str): path to the dataset directory

    Returns:
        str: extension of the files in the dataset directory
    """
    # make sure data_path directory exists
    assert is_valid_directory(data_path=data_path)

    # get the extension of the files in the dataset directory
    # assuming all files have the same extension
    for _, _, files in os.walk(data_path):
        for file in files:
            return str(os.path.splitext(file)[1])[1:]  # ignore the dot

    # if no files are found, raise an error
    raise FileNotFoundError(f"No files found under {data_path}.")


def get_loading_function(extension: str) -> Callable:
    """
    Returns the loading function corresponding to the file extension

    Args:
        extension (str): file extension

    Returns:
        Callable: loading function corresponding to the file extension
    """
    # get valid extensions and associated loading functions for cleaned images
    postgres_connection_string = get_postgres_connection_string()
    # pylint has a false positive when using psycopg 3's context managers
    # pylint: disable=not-context-manager
    with connect(postgres_connection_string) as conn:
        with conn.cursor() as curs:
            # get valid extensions for cleaned images
            curs.execute("SELECT format, load_func FROM file_formats;")
            res = curs.fetchall()
            if len(res) == 0:
                message = "No valid extensions found"
                logger.error_("%s", message)
                raise RuntimeError(message)
            # save valid extensions as set
            valid_extensions = {ext[0] for ext in res}
            # save loading functions as dictionary
            loading_funcs = {ext[0]: ext[1] for ext in res}
    # pylint: enable=not-context-manager

    # make sure the extension is valid
    assert extension in valid_extensions, (
        "Dataset is in invalid format. "
        f"Found {extension}, but valid extensions are {valid_extensions}"
    )

    # set the loading function based on the extension
    load_func_name = loading_funcs[extension]
    assert load_func_name in LOAD_FUNCS, (
        f"Could not find loading function for extension '{extension}'. "
        "Perhaps the loading function must be added "
        "to the constant LOAD_FUNCS dictionary "
        "in the constants.py script, and/or "
        "the entries in the database and "
        "dictionaries must be consistent."
    )

    load_func = LOAD_FUNCS[loading_funcs[extension]]

    return load_func


def np_save(array: ndarray, path: Path | str) -> None:
    """
    Provides a saving function for numpy arrays
    with a consistent interface with torch.save,
    i.e., swaps the order of the arguments

    Args:
        array (dict): NumPy array to save
        path (Path | str): path to save the array to
    """
    numpy_save(path, array)


def get_save_function(extension: str) -> Callable:
    """
    Returns the saving function corresponding to the file extension

    Args:
        extension (str): file extension, e.g., "npy", "pt"

    Returns:
        Callable: saving function corresponding to the file extension
    """
    # get valid extensions and associated saving functions for cleaned images
    postgres_connection_string = get_postgres_connection_string()
    # pylint has a false positive when using psycopg 3's context managers
    # pylint: disable=not-context-manager
    with connect(postgres_connection_string) as conn:
        with conn.cursor() as curs:
            # get valid extensions for cleaned images
            curs.execute("SELECT format, save_func FROM file_formats;")
            res = curs.fetchall()
            if len(res) == 0:
                message = "No valid extensions found"
                logger.error_("%s", message)
                raise RuntimeError(message)
            # save valid extensions as set
            valid_extensions = {ext[0] for ext in res}
            # save functions as dictionary
            save_funcs = {ext[0]: ext[1] for ext in res}
    # pylint: enable=not-context-manager

    # make sure the extension is valid
    assert extension in valid_extensions, (
        "Dataset is in invalid format. "
        f"Found {extension}, but valid extensions are {valid_extensions}"
    )

    # set the saving function based on the extension
    save_func_name = save_funcs[extension]
    assert save_func_name in SAVE_FUNCS, (
        f"Could not find saving function for extension '{extension}'. "
        "Perhaps the save function must be added "
        "to the constant SAVE_FUNCS and EXTENSION_TO_SAVE_FUNC"
        "dictionaries in the constants.py script, and/or "
        "the entries in the database and "
        "dictionaries must be consistent."
    )

    # exclude jpg and jpeg files for now
    if extension in ["jpg", "jpeg"]:
        message = "JPG and JPEG files are not supported for saving yet."
        logger.error_("%s", message)
        raise NotImplementedError(message)

    save_func = EXTENSION_TO_SAVE_FUNC[extension]

    # ignore type because MyPy says that save_func is a function
    # instead of a Callable, but a function is a Callable
    return save_func  # type: ignore


def get_device(
    verbose: bool = False, outside_logger: MaviraTrainLogger | None = None
) -> str:
    """
    Returns the string of the available accelerator type

    Args:
        verbose (bool, optional): whether or not to
            print available accelerators. Defaults to False.
        outside_logger (MaviraTrainLogger, optional): logger
            for reporting errors. Defaults to None.
            Must be provided if verbose is True.

    Returns:
        str: device string corresponding to available accelerator
    """
    assert (
        not verbose or outside_logger
    ), "Must provide a logger if verbose is True."

    if is_cuda_available():
        if verbose and outside_logger:
            outside_logger.info_("CUDA available")
        return "cuda"
    if verbose and outside_logger:
        outside_logger.info_("CUDA not available")

    if is_mps_available():
        if verbose and outside_logger:
            outside_logger.info_("MPS available")
        return "mps"
    if verbose and outside_logger:
        outside_logger.info_("MPS not available")

    if verbose and outside_logger:
        outside_logger.info_("Only CPU available")
    return "cpu"


def get_postgres_connection_string(
    database: str = "mavirafashiontrainingdb",
    host: str = "localhost",
    user: str = "mavira",
    password: str | None = ".env",
    passfile: str = "~/programs/postgresql/.pgpass",
    port: str = "5432",
) -> str:
    """
    Returns a connection string to connect to a PostgreSQL database
    using psycopg according to the provided parameters. Defaults to
    connecting to the "mavirafashiontrainingdb" database as user
    "mavira" hosted locally on port 5432.

    Args:
        database (str, optional): Name of database to connect to.
            Defaults to "mavirafashiontrainingdb".
        host (str, optional): Address of database host.
            Defaults to "localhost".
        user (str, optional): Name of user to connect as.
            Defaults to "mavira".
        password (str | None, optional): Password for user to be connected.
            Defaults to ".env", which loads a variable stored in
            the .env file of the project
            as "POSTGRESQL_USERMAVIRA_PASSWORD".
        passfile (str, optional): Path to the .pgpass file.
            Defaults to "~/programs/postgresql/.pgpass".
            Ignored if password is not None.
        port (str, optional): Port to connect to PostgreSQL on.
            Defaults to PostgreSQL's default of "5432".

    Returns:
        tuple[str]: a formatted connection string to connect with
            a PostgreSQL database using psycopg
    """
    # set up password or passfile
    if password is None:
        # if password is None, use the .pgpass file
        password_or_file = f"passfile={passfile}"
    elif password == ".env":
        # load default password for user "mavira" from .env file
        load_dotenv()
        password_or_file = (
            f'password={os.environ["POSTGRESQL_USERMAVIRA_PASSWORD"]}'
        )
    else:
        # otherwise, use the provided password
        password_or_file = f"password={password}"

    # format the connection string following PostgreSQL standards
    connection_string = (
        f"host={host} "
        f"port={port} "
        f"dbname={database} "
        f"user={user} "
        f"{password_or_file}"
    )

    return connection_string


def get_data_processing_job_id(new: bool = False) -> int:
    """
    Gets the most recent job ID from the database.
    If new is True, returns the next job ID,
    i.e., the most recent job ID plus 1.

    Returns:
        int: the most recent (next if new is True) job ID
    """
    job_id = -1  # placeholder for job_id
    postgres_connection_string = get_postgres_connection_string()
    # pylint has a false positive when using psycopg 3's context managers
    # pylint: disable=not-context-manager
    with connect(postgres_connection_string) as conn:
        with conn.cursor() as curs:
            curs.execute("SELECT MAX(id) FROM data_processing_jobs;")
            res = curs.fetchone()
            # ignore type because psycopg returns (None,) if no jobs
            # but Pylance expects just None
            # this happens because we are using the MAX function
            if res[0]:  # type: ignore
                job_id = res[0]  # type: ignore
            else:
                assert new, (
                    "No jobs found, but new is False. "
                    "This should only happen if no jobs have been run yet, "
                    "in which case new should be True."
                )
                # if no jobs have been run yet, start at 1
                job_id = 0
    # pylint: enable=not-context-manager

    # if getting a job ID for a job before it starts, increment the job ID
    if new:
        job_id += 1

    return job_id


def get_dataset_id(data_path: Path | str) -> int:
    """
    Gets the dataset ID from the database

    Args:
        data_path (Path | str): path to the dataset directory

    Returns:
        int: the dataset ID
    """
    dataset_id = -1  # placeholder for dataset_id
    postgres_connection_string = get_postgres_connection_string()
    # pylint has a false positive when using psycopg 3's context managers
    # pylint: disable=not-context-manager
    with connect(postgres_connection_string) as conn:
        with conn.cursor() as curs:
            curs.execute(
                "SELECT id FROM datasets WHERE path = %s;", (str(data_path),)
            )
            res = curs.fetchone()
            if res:
                dataset_id = res[0]
            else:
                # if no dataset is found, raise an error
                raise FileNotFoundError(f"No dataset found at {data_path}.")
    # pylint: enable=not-context-manager

    return dataset_id
