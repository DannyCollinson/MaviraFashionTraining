"""Script to run data processing pipeline"""

import argparse

from maviratrain.data.data_processing import run_full_processing_pipeline

parser = argparse.ArgumentParser(description="For running data pipeline")
parser.add_argument("data_path", type=str, help="Path to data to be processed")

args = parser.parse_args()  # get command line argument for data_path

# run processing pipeline on data at given path
result = run_full_processing_pipeline(args.data_path)
