""" Custom logging module for MaviraTrain. """

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from time import gmtime
from typing import Any


class UTCFormatter(logging.Formatter):
    """Custom formatter class for logging in UTC time."""

    converter = gmtime


def _get_logger(
    name: str,
    level: str = "debug",
    log_filename: Path | str | None = None,
    msg_formatter: str | None = None,
    rotation_params: tuple[int, int] | None = None,
    both: bool = True,
    console_level: str = "info",
) -> logging.Logger:
    """
    Returns a logger object with the specified parameters.
    A logger constructed here is a "MaviraTrainLogger".

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
    # get logger with specified name
    return_logger = logging.getLogger(name)

    # set the logging level if specified, default to DEBUG
    if level:
        return_logger.setLevel(level.upper())  # convert to uppercase
    else:
        return_logger.setLevel(logging.DEBUG)

    # split into cases because of MyPy not liking different handler types
    if log_filename:
        # create formatter
        if msg_formatter:
            # use specified format if provided
            formatter = UTCFormatter(msg_formatter)
        else:
            # use default format if not specified
            formatter = UTCFormatter(
                "%(asctime)s+00:00 - %(name)s - %(levelname)s - %(message)s"
            )

        if rotation_params:
            # create rotating file handler if rotation_params is specified
            rotating_file_handler = RotatingFileHandler(
                log_filename,
                maxBytes=rotation_params[0],
                backupCount=rotation_params[1],
                encoding="utf-8",
            )

            # connect formatter, logger, and file handler
            rotating_file_handler.setFormatter(formatter)
            return_logger.addHandler(rotating_file_handler)
        else:
            # create regular file handler if no rotation_params is specified
            file_handler = logging.FileHandler(log_filename, encoding="utf-8")

            # connect formatter, logger, and file handler
            file_handler.setFormatter(formatter)
            return_logger.addHandler(file_handler)

    if both or not log_filename:
        # create formatter
        if msg_formatter:
            # use specified format if provided
            formatter = UTCFormatter(msg_formatter)
        else:
            # use default format if not specified
            formatter = UTCFormatter(
                "%(asctime)s - %(message)s",
                datefmt="%H:%M:%S",
            )

        # create console handler if no log_filename is specified or both wanted
        console_handler = logging.StreamHandler()

        # set the logging level for the console handler if both is true
        if both:
            console_handler.setLevel(console_level.upper())

        # connect formatter, logger, and console handler
        console_handler.setFormatter(formatter)
        return_logger.addHandler(console_handler)

    return return_logger


class MaviraTrainLogger(logging.Logger):
    """
    Custom logger class for MaviraTrain
    """

    def __init__(
        self,
        name: str,
        level: str = "debug",
        log_filename: Path | str | None = None,
        msg_formatter: str | None = None,
        rotation_params: tuple[int, int] | None = None,
        both: bool = True,
        console_level: str = "info",
    ) -> None:
        """
        Initializes the logger with the specified parameters.

        Args:
            name (str): the name of the logger
            level (str, optional): the logging level. Defaults to DEBUG.
                If both is true, this will be
                the level for the file logger.
            log_filename (Path | str, optional): the path to
                the log file. Defaults to None,
                which means the logger outputs to stderror.
            msg_format (str, optional): the format of the log messages.
                Defaults to
                YYYY-MM-DDTHH:MM:SS+00+00 - name - LEVEL - msg.
                Note that time is in UTC regardless of specified format.
            rotation_params: tuple[int, int] -- the rotation parameters
                for the log file. Defaults to None,
                which means no rotation. The first integer
                is the maximum size of the log file in
                bytes before rotation, and the second integer is the
                number of backup log files. Will only be used
                if log_filename is specified.
            both (bool, optional): whether to log to both
                file and console. Defaults to True.
            console_level (str, optional): the logging level for
                the console logger if both is true. Defaults to INFO.
        """
        # call parent constructor
        super().__init__(name)

        # set custom attributes
        self.mavira_name = name
        self.string_level = level
        self.log_filename = log_filename
        self.msg_formatter = msg_formatter
        self.rotation_params = rotation_params
        self.both = both
        self.console_level = console_level

        self.logger = _get_logger(
            name,
            level,
            log_filename,
            msg_formatter,
            rotation_params,
            both,
            console_level,
        )

    def set_level(self, level: str) -> None:
        """
        Sets the logging level for the logger

        Args:
            level (str): the logging level
        """
        self.string_level = level
        self.logger = _get_logger(
            self.mavira_name,
            level,
            self.log_filename,
            self.msg_formatter,
            self.rotation_params,
            self.both,
            self.console_level,
        )

    def debug_(self, message: str, *args: Any) -> None:
        """
        Logs a debug message. The underscore is to avoid
        shadowing the built-in debug method.

        Args:
            message (str): the message to log
            args (tuple): the arguments to format the message
        """
        self.logger.debug(message, *args)

    def info_(self, message: str, *args: Any) -> None:
        """
        Logs an info message. The underscore is to avoid
        shadowing the built-in info method.

        Args:
            message (str): the message to log
            args (tuple): the arguments to format the message
        """
        self.logger.info(message, *args)

    def warning_(self, message: str, *args: Any) -> None:
        """
        Logs a warning message. The underscore is to avoid
        shadowing the built-in warning method.

        Args:
            message (str): the message to log
            args (tuple): the arguments to format the message
        """
        self.logger.warning(message, *args)

    def error_(self, message: str, *args: Any) -> None:
        """
        Logs an error message. The underscore is to avoid
        shadowing the built-in error method.

        Args:
            message (str): the message to log
            args (tuple): the arguments to format the message
        """
        self.logger.error(message, *args)

    def critical_(self, message: str, *args: Any) -> None:
        """
        Logs a critical message. The underscore is to avoid
        shadowing the built-in critical method.

        Args:
            message (str): the message to log
            args (tuple): the arguments to format the message
        """
        self.logger.critical(message, *args)

    def set_log_filename(self, log_filename: Path | str | None) -> None:
        """
        Sets the log filename for the logger

        Args:
            log_filename (Path | str | None): the path to the log file
        """
        self.log_filename = log_filename
        self.logger = _get_logger(
            self.mavira_name,
            self.string_level,
            log_filename,
            self.msg_formatter,
            self.rotation_params,
            self.both,
            self.console_level,
        )

    def set_msg_formatter(self, msg_formatter: str | None) -> None:
        """
        Sets the message formatter for the logger

        Args:
            msg_formatter (str | None): the format of the log messages
        """
        self.msg_formatter = msg_formatter
        self.logger = _get_logger(
            self.mavira_name,
            self.string_level,
            self.log_filename,
            msg_formatter,
            self.rotation_params,
            self.both,
            self.console_level,
        )

    def set_rotation_params(
        self, rotation_params: tuple[int, int] | None
    ) -> None:
        """
        Sets the rotation parameters for the logger

        Args:
            rotation_params (tuple[int, int] | None): the new rotation
                parameters for the log file
        """
        self.rotation_params = rotation_params
        self.logger = _get_logger(
            self.mavira_name,
            self.string_level,
            self.log_filename,
            self.msg_formatter,
            rotation_params,
            self.both,
            self.console_level,
        )

    def set_both(self, both: bool) -> None:
        """
        Sets whether to log to both file and console

        Args:
            both (bool): whether to log to both file and console
        """
        self.both = both
        self.logger = _get_logger(
            self.mavira_name,
            self.string_level,
            self.log_filename,
            self.msg_formatter,
            self.rotation_params,
            both,
            self.console_level,
        )

    def set_console_level(self, console_level: str) -> None:
        """
        Sets the logging level for the console logger

        Args:
            console_level (str): the logging level for the console logger
        """
        self.console_level = console_level
        self.logger = _get_logger(
            self.mavira_name,
            self.string_level,
            self.log_filename,
            self.msg_formatter,
            self.rotation_params,
            self.both,
            console_level,
        )

    def close_logger(self) -> None:
        """
        Closes all handlers and removes them from the logger
        """
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)
