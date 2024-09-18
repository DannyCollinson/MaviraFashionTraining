"""Common utility functions"""

import datetime
import os

from dotenv import load_dotenv
from psycopg2 import connect
from psycopg2.extensions import connection, cursor
from torch.backends.mps import is_available as is_mps_available
from torch.cuda import is_available as is_cuda_available


def get_log_time() -> str:
    """
    Returns the current UTC datetime in ISO format truncated to seconds

    Returns:
        str -- the current datetime
    """
    return datetime.datetime.now(tz=datetime.timezone.utc).isoformat(
        timespec="seconds"
    )


def get_file_date() -> str:
    """
    Returns the current UTC date in ISO format

    Returns:
        str -- the current date
    """
    return datetime.datetime.now(tz=datetime.timezone.utc).isoformat(
        timespec="seconds"
    )[:10]


def is_valid_directory(data_path) -> bool:
    """
    Verifies that data_path points to a directory

    Arguments:
        data_path {_type_} -- path to check for directory

    Returns:
        bool -- True if directory exists, false otherwise
    """
    # make sure data directory exists
    res = os.path.isdir(data_path)
    if not res:
        print(f"Expected data_path to point to a directory. Got {data_path}.")
    return res


def get_device(verbose: bool = False) -> str:
    """
    Returns the string of the available accelerator

    Returns:
        str: device string corresponding to available accelerator
    """
    if is_cuda_available():
        if verbose:
            print("CUDA available")
        return "cuda"
    if verbose:
        print("CUDA not available")

    if is_mps_available():
        if verbose:
            print("MPS available")
        return "mps"
    if verbose:
        print("MPS not available")

    if verbose:
        print("Only CPU available")
    return "cpu"


def connect_postgres(
    database: str = "mavirafashiontrainingdb",
    host: str = "localhost",
    user: str = "mavira",
    password: str = ".env",
    port: str = "5432",
    autocommit: bool = False,
) -> tuple[connection, cursor]:
    """
    Returns a connection to a PostgreSQL database configured according to the
    provided parameters as well as a cursor for executing SQL commands.
    By default connects locally to "mavirafashiontrainingdb" as user "mavira".

    Args:
        database (str, optional): Name of database to connect to.
            Defaults to "mavirafashiontrainingdb".
        host (str, optional): Address of database host.
            Defaults to "localhost".
        user (str, optional): Name of user to connect as. Defaults to "mavira".
        password (str, optional): Password for user to be connected.
            Defaults to ".env", which loads a variable stored in the .env file
            of the project as "USERMAVIRA_MAVIRAFASHIONTRAININGDB_PASSWORD".
        port (str, optional): Port to connect to PostgreSQL on.
            Defaults to PostgreSQL's default of "5432".
        autocommit (bool, optional): Whether to set the connection to
            autocommit mode or not. Defaults to False.

    Returns:
        tuple[connection, cursor]: the connection and cursor psycopg2 objects.
            Remember to close cursor and connection when finished with use.
    """
    # load default password for user "mavira" from .env file if needed
    if password == ".env":
        load_dotenv()
        password = os.environ["USERMAVIRA_MAVIRAFASHIONTRAININGDB_PASSWORD"]

    conn = connect(  # connect to database
        database=database, host=host, user=user, password=password, port=port
    )
    conn.autocommit = autocommit  # set autocommit mode

    curs = conn.cursor()  # create cursor

    return conn, curs  # remember to close cursor and connection when finished
