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
mkdir -p ./checkpoints/classifier/models
mkdir -p ./checkpoints/classifier/optimizers
mkdir -p ./checkpoints/classifier/schedulers
mkdir -p ./checkpoints/classifier/scalers
mkdir -p ./checkpoints/mae
mkdir -p ./checkpoints/mae/models
mkdir -p ./checkpoints/mae/optimizers
mkdir -p ./checkpoints/mae/schedulers
mkdir -p ./checkpoints/mae/scalers
mkdir -p ./logs
mkdir -p ./logs/data_processing
mkdir -p ./logs/train_runs
mkdir -p ./logs/train_runs/classifier
mkdir -p ./logs/train_runs/mae
mkdir -p ./logs/archive
mkdir -p ./logs/archive/data_processing
mkdir -p ./logs/archive/train_runs
mkdir -p ./logs/archive/train_runs/classifier
mkdir -p ./logs/archive/train_runs/mae

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
