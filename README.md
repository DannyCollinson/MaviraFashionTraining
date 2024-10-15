# Important
This is a copy of the Mavira FashionTraining repository written solely by Danny Collinson for the purposes of providing a personal code sample. No permissions are granted to anyone outside of Mavira except for those outlined in GitHub's Terms of Use.

This repository is a work in progress and will be updated periodically as changes to the parent repository are made.


# Training your Personal Fashion Stylist

The code here is for training the models used for the AI fashion assistant. It includes the `maviratrain` package.

## Getting Started

### Short Version
With the project repository cloned from GitHub (https://github.com/MaviraAI/FashionTraining.git) and `micromamba` installed, run the following command from the project directory. Then install the following VSCode Extensions as well as any others that you prefer to use and that do not conflict with the following: Python, Pylance, Pylint, Jupyter, Black Formatter, autoDocstring, isort, Mypy Type Checker. Finally, follow the __PostgreSQL Setup__ section below. If you need help at any point, try taking a look at the Long Version below and ask Danny (dannycollinson12@gmail.com or danny@maviraai.com) if you still need help.
```
micromamba create --yes -n maviratrain \
        fsspec pytest ipykernel seaborn python-dotenv sympy=1.13.1 pytorch torchvision pytorch-cuda=12.4 \
        -c pytorch-nightly -c nvidia --channel-priority=flexible \
    && micromamba activate maviratrain \
    && micromamba clean --all --yes \
    && python -m ipykernel install --user --name maviratrain \
    && pip install psycopg[binary] \
    && python -m pip install -e maviratrain
```


### Long Version
Follow the steps below in order and you should be all set up. If you need help at any point, ask Danny (dannycollinson12@gmail.com or danny@maviraai.com).
1. Make sure that `micromamba` is installed. If not, follow https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html to install.
2. If you do not already have a GitHub account, ask Danny (dannycollinson12@gmail.com or danny@maviraai.com) to help you set one up as a member of the MaviraAI organization.
3. Make sure that Git is installed. If not, follow https://git-scm.com/downloads to install. If you need to configure Git or do not already have a Personal Access Token (PAT) to enter as the GitHub password for the next step, then follow the directions in the __Git Setup__ section below.
4. Clone this repository by running `git clone https://github.com/MaviraAI/FashionTraining.git` from your directory for Mavira projects.
5. Run the following command from the project directory and confirm by pressing `Enter` when prompted to install necessary dependencies via `micromamba`.
```
micromamba create -n maviratrain \
    fsspec pytest ipykernel seaborn python-dotenv sympy=1.13.1 pytorch torchvision pytorch-cuda=12.4 \
    -c pytorch-nightly -c nvidia --channel-priority=flexible
```
6. Run `micromamba activate maviratrain` to activate the environment.
7. Run `micromamba clean --all --yes` to clean up the `micromamba` installation.
8. Run `python -m ipykernel install --user --name maviratrain` to create an IPython kernel for use in notebooks.
9. Run `pip install psycopg[binary]` to install `psycopg` for interacting with our PostgreSQL database from Python.
10. Run `python -m pip install -e maviratrain` to install an editable version of our training package that updates as you make changes.
11. Run `micromamba env export --no-build` and make sure that it looks like the `env.yml` file in the main project directory.
12. Install the following VSCode extensions: Python, Pylance, Pylint, Jupyter, Black Formatter, MyPy, autoDocstring, isort, Mypy Type Checker.
13. Install other VSCode extensions that you prefer to use and that do not conflict with those already installed.
14. Follow the __PostgreSQL Setup__ section below.


### Git Setup
If you have not configured Git, then run the commands `git config --global user.name "<Your Name>"` and `git config --global user.email "<your-email@example.com>"` to set up your configurations. Next, log into https://github.com, click your icon in the top right, select Settings, and then select Developer Settings at the bottom of the menu on the left. Next, open the Personal Access Tokens dropdown and select Fine-grained tokens. Give your token a descriptive name, set a Custom expiration date of 1 year, select "All repositories", give yourself all permissions, and finally hit Generate token. Save the generated token somewhere safe and use it as the password when cloning the repository.

You can also set up Git credential storage so that you do not have to enter your username and PAT every time. Instructions vary by system, so check https://git-scm.com/book/en/v2/Git-Tools-Credential-Storage and ask Danny (dannycollinson12@gmail.com or danny@maviraai.com) if you need help.


### PostgreSQL Setup
We are currently running PostgreSQL 16.4 locally, so each person must set up their own local installation. To do so, follow the steps below. As always, ask Danny (dannycollinson12@gmail.com or danny@maviraai.com) if you need help.
1. Go to https://www.postgresql.org/download/ and select your operating system to get the correct installation instructions. If running Windows, we recommend using WSL and selecting the corresponding Linux distribution, but if not, you must download an installer. If running a Debian/Ubuntu-based distribution, you can install the current version (16.4) using the command `apt install postgresql`. If running macOS, then assuming Homebrew is installed (if not, follow https://brew.sh/), run `brew install postgresql@16`. After installation is complete, run `psql --version` to check what version of PostgreSQL is installed; contact Danny (dannycollinson12@gmail.com or danny@maviraai.com) if the version number is not 16.4.
2. Run the command `sudo passwd postgres` to create a password for the 'postgres' poweruser. Save this password for the future.
3. Run `sudo su - postgres` to switch to the 'postgres' user. Next, run `initdb -D /var/lib/pgsql/data` to initialize PostgreSQL to be able to store data on your system. Then, run `exit` to switch back from the 'postgres' user to your own user.
4. Run `sudo systemctl start postgresql` to start the local PostgreSQL server, then `sudo systemctl status postgresql` and `sudo systemctl enable postgresql` to finish setting up the server.
5. Switch back to the 'postgres' user using `sudo su - postgres` and log in to psql by running `psql`.
6. From within the psql console, run `CREATE ROLE mavira WITH CREATEDB LOGIN PASSWORD '<POSTGRESQL_USERMAVIRA_PASSWORD value from .env file>';`, where you replace `<POSTGRESQL_USERMAVIRA_PASSWORD from .env file>` with the value assigned to the `POSTGRESQL_USERMAVIRA_PASSWORD` variable in the `.env` file (contact Danny (dannycollinson12@gmail.com or danny@maviraai.com) if you do not have the `.env` file.) to create a new user named 'mavira'.
7. Run `GRANT ALL PRIVILEGES ON DATABASE template0 TO mavira;` to enable to the 'mavira' user to connect to the `template0` database.
8. Run `\q` to quit psql and then run `exit` to log out as the 'postgres' user.
9. Run `psql -U mavira template0 -h localhost` to log in as the 'mavira' user on the template database.
10. Run `CREATE DATABASE mavirafashiontrainingdb;` and then `GRANT ALL PRIVILEGES ON DATABASE mavirafashiontrainingdb TO mavira;` to make sure that the 'mavira' user can access the new database.
11. Run `\q` to log out of the connection to the template database. Confirm that you can connect to the new database using `psql -U mavira mavirafashiontrainingdb -h localhost`, then disconnect with `\q`.
12. From the project directory, run `bash posgres/set_up_tables.sh` to create the necessary tables in the `mavirafashiontrainingdb` database.
13. Open `data_processing.ipynb` in the `notebooks` folder of the project directory. Assuming you have a dataset downloaded from the Mavira Google Drive and unzipped into a subfolder of the `data` folder of the project directory (You can unzip data if needed by running `python unzip_data.py path/to/zipped/data path/to/place/zipped/data regex` from the project directory, where `regex` is a regular expression to match the zipped files, typically `*.zip`.), then you can follow the directions under 'Dataset Registration' to register your dataset in the mavirafashiontrainingdb database, afterwhich you are all set up to begin data processing!


## Reinstalling Environment
To reinstall the environment from scratch to install the latest PyTorch version or install new dependencies, follow these steps:
1. If the `maviratrain` environment is not already active, run `micromamba activate maviratrain`.
2. Run `jupyter kernelspec uninstall maviratrain` and enter `y` to confirm and uninstall the associated Jupyter kernel.
3. Run `micromamba deactivate` to deactivate the environment.
4. Run `micromamba env remove --yes -n maviratrain` to remove the existing environment.
5. Run the command from the __Short Version__ of the __Getting Started__ section above or follow steps 4-9 in the __Long Version__ section.


## Help
Ask Danny (dannycollinson12@gmail.com or danny@maviraai.com) if you have any questions.
For now, we are each running everything locally until we have to scale up and run things in the cloud.
