#!/bin/bash

# set up the tables in the mavirafashiontrainingdb PostgreSQL database
# according to the specifications in the files found in the tables directory
# by using the psql commands in this file

# set up the normalization_methods table
psql -U mavira mavirafashiontrainingdb -h localhost \
    -f postgres/tables/normalization_methods.sql

# set up the valid_file_formats table
psql -U mavira mavirafashiontrainingdb -h localhost \
    -f postgres/tables/valid_file_formats.sql

# set up the datasets table
psql -U mavira mavirafashiontrainingdb -h localhost \
    -f postgres/tables/datasets.sql

# set up the data_processing_jobs table
psql -U mavira mavirafashiontrainingdb -h localhost \
    -f postgres/tables/data_processing_jobs.sql

# set up the optimizers table
psql -U mavira mavirafashiontrainingdb -h localhost \
    -f postgres/tables/optimizers.sql

# set up the lr_schedulers table
psql -U mavira mavirafashiontrainingdb -h localhost \
    -f postgres/tables/lr_schedulers.sql

# set up the loss_functions table
psql -U mavira mavirafashiontrainingdb -h localhost \
    -f postgres/tables/loss_functions.sql

# set up the checkpoints table
psql -U mavira mavirafashiontrainingdb -h localhost \
    -f postgres/tables/checkpoints.sql

# set up the train_runs table
psql -U mavira mavirafashiontrainingdb -h localhost \
    -f postgres/tables/train_runs.sql

# add initial data to some tables
psql -U mavira mavirafashiontrainingdb -h localhost \
    -f postgres/tables/initial_setup.sql