#!/bin/bash

# make sure the script is run from the main project directory
cd ~/mavira/FashionTraining

# initialize the tables in the postgresql database
echo "Initializing tables in the PostgreSQL database..."
bash ./postgres/set_up_tables.sh

# set up the data, checkpoints, and logs directories
echo "Setting up data, checkpoints, and logs directories..."
mkdir -p ./data
mkdir -p ./data/classifier
mkdir -p ./data/mae
mkdir -p ./checkpoints
mkdir -p ./checkpoints/classifier
mkdir -p ./checkpoints/mae
mkdir -p ./logs
mkdir -p ./logs/archive
mkdir -p ./logs/archive/data_processing
mkdir -p ./logs/archive/train_runs
mkdir -p ./logs/data_processing
mkdir -p ./logs/train_runs

# run tests
echo "Running tests..."
pytest ./tests

# test the PyTorch install
# make sure the Python environment is activated
echo "Testing PyTorch install..."
eval "$(micromamba shell hook --shell bash)"
echo "Available accelerator devices:"
# run the test script
python3 ./scripts/test_pytorch_install.py
