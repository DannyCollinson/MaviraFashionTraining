"""Script to quickly reset tests. Modify as needed."""

from maviratrain.utils.cleanup import (
    clear_data_processing_logs,
    reset_test,
    reset_test_resized,
    reset_test_resized_split,
)

clear_data_processing_logs()
reset_test()
reset_test_resized()
reset_test_resized_split()
