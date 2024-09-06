"""
Utility functions for running quick tests in development.
Modify as needed
"""

import os


def clear_data_processing_logs() -> None:
    """
    Clears the logs for all data processing tasks
    """
    os.system("rm logs/data_processing/subdir_cleaning*.log")
    os.system("rm logs/data_processing/filename_cleaning*.log*")
    os.system("rm logs/data_processing/data_name_cleaning*.log")
    os.system("rm logs/data_processing/image_resizing*.log")
    os.system("rm logs/data_processing/train_val_test*.log")
    os.system("rm logs/data_processing/normalize_data*.log")
    os.system("rm logs/data_processing/run_full_processing_pipeline*.log")


def reset_test():
    """
    Resets the test/ directory to be a copy of the test_copy/ directory
    """
    os.system("rm -r /workspaces/FashionTraining/data/classification_test")
    os.system(
        "cp -r /workspaces/FashionTraining/data/classification_test_copy"
        "/workspaces/FashionTraining/data/classification_test"
    )


def reset_test_resized():
    """
    Deletes the test-resized_224x224-YYYY-MM-DD/ directory
    """
    os.system(
        "rm -r /workspaces/FashionTraining/data/"
        "classification_test-resized_224x224-"
        "[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]"
    )


def reset_test_resized_split():
    """
    Deletes the test-resized_224x224-YYYY-MM-DD-split-YYYY-MM-DD/ directory
    """
    os.system(
        "rm -r /workspaces/FashionTraining/data/"
        "classification_test-resized_224x224-"
        "[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]"
        "-split-[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]"
    )
