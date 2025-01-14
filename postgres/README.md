## Information about the PostgreSQL database used for logging info about datasets, processing runs, training runs, checkpoints, etc.

Running `set_up_tables.sh` initializes all of the tables in the `mavirafashiontrainingdb` database.

Each user creates their own local database to store information about their resources. In the future, we may set up a centralized database with records from all users, but this is not planned for implementation in the near future.
