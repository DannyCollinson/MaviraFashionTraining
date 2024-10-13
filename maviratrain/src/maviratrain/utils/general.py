""" Common utility functions. """

import datetime
import os
from collections.abc import Callable
from pathlib import Path

from dotenv import load_dotenv
from psycopg import connect
from torch.backends.mps import is_available as is_mps_available
from torch.cuda import is_available as is_cuda_available

from .constants import LOAD_FUNCS
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
    data_path: Path | str, outside_logger: MaviraTrainLogger | None = None
) -> bool:
    """
    Verifies that data_path points to a subdirectory of the
    "data" directory

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
            return str(os.path.splitext(file)[1])

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
            curs.execute(
                "SELECT format, loading_func FROM valid_file_formats;"
            )
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
        "to the constant LOAD_FUNCS dictionary in this script, "
        "and/or the entries in the database "
        "and in dictionaries must be consistent."
    )

    load_func = LOAD_FUNCS[loading_funcs[extension]]

    return load_func


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
    password: str = ".env",
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
        password (str, optional): Password for user to be connected.
            Defaults to ".env", which loads a variable stored in
            the .env file of the project
            as "POSTGRESQL_USERMAVIRA_PASSWORD".
        port (str, optional): Port to connect to PostgreSQL on.
            Defaults to PostgreSQL's default of "5432".

    Returns:
        tuple[str]: a formatted connection string to connect with
            a PostgreSQL database using psycopg
    """
    # load default password for user "mavira" from .env file if needed
    if password == ".env":
        load_dotenv()
        password = os.environ["POSTGRESQL_USERMAVIRA_PASSWORD"]

    # format the connection string following PostgreSQL standards
    connection_string = (
        f"host={host} "
        f"port={port} "
        f"dbname={database} "
        f"user={user} "
        f"password={password}"
    )

    return connection_string


def get_data_processing_job_id() -> int:
    """
    Gets the next job ID from the database,
    i.e., the most recent job ID + 1

    Returns:
        int: the most recent job ID
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
                job_id = res[0] + 1  # type: ignore
            else:
                # if no jobs have been run yet, start at 1
                job_id = 1
    # pylint: enable=not-context-manager

    return job_id
