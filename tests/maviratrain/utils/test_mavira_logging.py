""" Tests for the maviratrain.utils.mavira_logging module. """

import os

from maviratrain.utils.general import get_logger


def test_placeholder():
    """
    A placeholder test
    """
    assert True


def test_logger():
    """
    Test the logger
    """
    logger = get_logger(
        "tests.utils.test_mavira_logging",
        log_filename="test.log",
        rotation_params=(1000000, 1000),
        both=True,
        console_level="DEBUG",
    )

    logger.debug_("Debug message")
    logger.info_("Info message")
    logger.warning_("Warning message")
    logger.error_("Error message")
    logger.critical_("Critical message")

    with open("test.log", "r", encoding="utf-8") as f:
        log_content = f.read()

    assert (
        " - tests.utils.test_mavira_logging - DEBUG - Debug message"
        in log_content
    )
    assert (
        " - tests.utils.test_mavira_logging - INFO - Info message"
        in log_content
    )
    assert (
        " - tests.utils.test_mavira_logging - WARNING - Warning message"
        in log_content
    )
    assert (
        " - tests.utils.test_mavira_logging - ERROR - Error message"
        in log_content
    )
    assert (
        " - tests.utils.test_mavira_logging - CRITICAL - Critical message"
        in log_content
    )

    os.remove("test.log")
