# Training your Personal Fashion Stylist

The code here is for training the models used for the AI fashion assistant. It includes the `maviratrain` package. The guide below will guide you in setting up your system and environment to get started training your own models.

Most of the instructions here focus on systems running Linux, and particularly Debian/Ubuntu distributions, but much of the following should work on other distributions and systems like MacOS and Windows by following the system-specific instructions at the links provided when applicable and inserting equivalent package manager and CLI commands as needed. If using Windows, you can use Git Bash (https://gitforwindows.org/), but we recommend using WSL 2 instead (https://learn.microsoft.com/en-us/windows/wsl/install).

If you need help at any point, you can contact Danny Collinson at dannycollinson12@gmail.com or danny@maviraai.com for support.


## Getting Set Up

Follow the steps below in order and you should be all set up. If you need help at any point, contact Danny.
0. If starting from scratch by creating a new Google Cloud Compute Engine VM, follow the __Instance Creation__ and __Connecting for the First Time__ sections of the __Google Cloud Compute Engine VM Setup__ section below, and return here when finished.
1. If your machine has an NVIDIA GPU but does not have the appropriate NVIDIA drivers or CUDA Toolkit installed, follow the __GPU Setup__ section below, and return here when finished.
2. Run `mkdir programs` to create a directory for storing info for some of the different programs that we will be using.
3. If you do not already have a GitHub account that is a member of the MaviraAI organization, contact Danny to help you set up a new account or connect an existing account to the MaviraAI organization.
4. If `git` is not installed, (If running `git --help` throws an error, then it is likely not installed), follow the system-specific instructions at https://git-scm.com/downloads to install it. For Debian/Ubuntu systems, this is just `sudo apt install git`. To configure Git and to create a Personal Access Token (PAT) if needed for connecting to Mavira on GitHub, follow the directions in the __Git Setup__ section below, and return here when finished.
5. From your home directory, use `mkdir mavira && cd mavira` to create and move into a new directory for Mavira projects.
6. Clone this repository by running `git clone https://github.com/MaviraAI/FashionTraining.git` from your directory for Mavira projects. Enter your GitHub username and then the PAT that you created during __Git Setup__ as the password.
7. If `micromamba` is not installed (If running `micromamba --help` throws an error, then it is likely not installed), follow https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html to install it. For Linux systems, this is just `"${SHELL}" <(curl -L micro.mamba.pm/install.sh)`. Except for the `MAMBA_ROOT_PREFIX` option, the default options are fine. For `MAMBA_ROOT_PREFIX`, use `~/programs/micromamba` when prompted instead of the default `~/micromamba`.
8. Follow the __Python Environment Setup__ section below, and return here when finished.
9. Follow the __PostgreSQL Setup__ section below to set up the local PostgreSQL database used to track datasets, processing jobs, and experiments, and return here when finished.
10. Install the following VS Code extensions: Python, Pylance, Pylint, Jupyter, Black Formatter, Mypy Type Checker, autoDocstring, and isort.
11. Optionally, install the VS Code extensions SQLTools and SQLTools PostgreSQL/Cockroach Driver to interact with the PostgreSQL database through a VS Code-based interface.
12. Install other VS Code extensions that you prefer to use and that do not conflict with those already installed; e.g., GitHub Copilot.
13. Proceed to the __Getting Started__ section below.


### Python Environment Setup
The following instructions  will help you set up the correct Python environment. The end of this section contains an all-in-one command for creating, activating, and setting up the environment that will work for most systems.
1. Run `micromamba create -n maviratrain && micromamba activate maviratrain` to create and activate a Python environment for training.
2. Run `micromamba install fsspec pytest pytest-cov ipykernel seaborn python-dotenv sympy numpy` and press `Enter` when prompted to install many of the necessary packages.
3. Go to https://pytorch.org/ to get the command to run for your system for installing the PyTorch Preview release (You can remove `torchaudio` from the command). For a Linux or Windows machine using CUDA version 12.8, the command is `pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128`, and for different CUDA versions, just change the last three characters.
4. Run `pip install psycopg[binary] torcheval pytorch-focalloss` to install `psycopg` for interacting with our PostgreSQL database from Python and `torcheval` for computing training metrics.
5. Run `micromamba clean --all --yes` to clean up after the `micromamba` installation process.
6. Run `python -m ipykernel install --user --name maviratrain` to create an IPython kernel for use in notebooks.
7. Run `pip install -e maviratrain` to install an editable version of our training package that updates as you make changes.
8. Run `pip check` to have `pip` look for any broken dependencies in the `maviratrain` environment. If there are, resolve them if the solution is obvious, or contact Danny for support.

As an alternative to the above, you can run a variation of the following command that depends on your system. Most notably, make sure that the fourth line of the command is correct for your system (You can go to https://pytorch.org/ and input your details to get the right command for you):
```
micromamba create -n maviratrain \
&& micromamba activate maviratrain \
&& micromamba install --yes fsspec pytest pytest-cov ipykernel seaborn python-dotenv sympy numpy \
&& pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128 \
&& pip install psycopg[binary] torcheval pytorch-focalloss \
&& micromamba clean --all --yes \
&& python -m ipykernel install --user --name maviratrain \
&& pip install -e maviratrain \
&& pip check
```


### Git Setup
If you need to configure Git, run the commands `git config --global user.name "<Your Name>"` and `git config --global user.email "<your-email@example.com>"`—replacing `<Your Name>` and `<your-email@example.com>` with your name and email associated with the Mavira GitHub organization, respectively, and making sure to leave all full quotation marks in place (e.g., "Danny Collinson")—to set up your configuration.

To create a GitHub Personal Access Token (PAT) to access Mavira projects, log into https://github.com, click your icon in the top right, select Settings, and then select Developer Settings at the bottom of the menu on the left. Next, open the Personal Access Tokens dropdown and select Fine-grained tokens. Give your token a descriptive name, set a Custom expiration date of 1 year, select "All repositories", give yourself all permissions, and finally hit Generate token. Save the generated token somewhere safe (like a password manager) and use it as the password when cloning the repository.

You can also set up Git credential storage so that you do not have to enter your username and PAT every time. Instructions vary by system, so check https://git-scm.com/book/en/v2/Git-Tools-Credential-Storage and contact Danny if you need support.


### GPU Setup
If you are training locally, you likely already have a driver installed. However, you should still follow along with this guide to ensure that your driver is compatible with the needed CUDA version for the version of PyTorch that you want to use.

We will be installing the CUDA Toolkit, which will automatically install the right compatible driver.

If using Ops Agent on a Google Cloud VM, check the note at the following link: https://cloud.google.com/monitoring/agent/ops-agent/configuration?authuser=2#receiver-nvml-metrics. This is not something to worry about for our default configuration, but it is somthing to be aware of.

1. Run `sudo apt install linux-headers-$(uname -r) build-essential dkms curl software-properties-common` to make sure that the required kernel headers and tools are installed before driver installation.
2. Run `sudo systemctl stop gdm3` to stop any potential conflicting processes.
3. Run `echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/ /" | sudo tee /etc/apt/sources.list.d/cuda.list` to add the NVIDIA repository.
4. Run `curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/<OS/architecture/file> | sudo gpg --dearmor -o /usr/share/keyrings/cuda-archive-keyring.gpg`, where `<OS/architecture/file>` is replaced by the correct info for your system to download the NVIDIA GPG key and add it the keyring directory. For Debian 12 on x86_64, this is `debian12/x86_64/3bf863cc.pub`.
5. Run `sudo apt update && sudo apt upgrade` to update and upgrade system packages.
6. Go to https://pytorch.org/get-started/locally/ to check the versions
7. Run `sudo apt install cuda-<major-minor>`, where `<major-minor>` are the major and minor versions of the CUDA Toolkit that you want to install; e.g., run `sudo apt install cuda-12-6` to install version 12.6. You may have to try several versions to get it to work, depending on which packages are available in the NVIDIA repository and the system you are using. We recommend starting from the most recent CUDA version supported by PyTorch and working backwards until one is available.
8. Run `nvidia-smi` to confirm that the driver and toolkit were installed correctly. The versions listed in the output may differ from what you specified in the install (They will likely be higher, if so), but this should not cause problems. If you find that it does, contact Danny for support. They may also change when setting up or updating the python environment or upgrading the system, but this also did not cause issues. If you find that it does, contact Danny for support.
9. Replace `<major.minor>` in the following two commands with the major and minor versions of the CUDA Toolkit that you want to install, and then run them: `echo 'export PATH=/usr/local/cuda-<major.minor>/bin:$PATH' >> ~/.bashrc` and `echo 'export LD_LIBRARY_PATH=/usr/local/cuda-<major.minor>/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc` to add the needed environment variables.


### PostgreSQL Setup
Each machine used for training runs a local PostgreSQL server to keep track of our datasets, processing jobs, and experiments. To set up the installation, follow the steps below. As always, contact Danny if you need support.
1. Go to https://www.postgresql.org/download/ and select your operating system to get the correct installation instructions. If running Windows, we recommend using WSL and selecting the corresponding Linux distribution, but if not, you must download an installer. If running a Debian/Ubuntu-based distribution, you will likely have to configure the PostgreSQL repository to get the latest version. This is done by running `sudo apt install postgresql-common`, followed by `sudo /usr/share/postgresql-common/pgdg/apt.postgresql.org.sh`, and finally `sudo apt update`. You can then run `sudo apt install postgresql` to install the latest version. After installation is complete, run `psql --version` to check what version of PostgreSQL is installed. As of this writing (December 2024), this should be version 17.
2. In the root directory of the training repository, create a `.env` file by running `nano .env`; this file is already included in the `.gitignore` file, so it will not be included in version control. Add the two lines below to the file, replacing `<password>` and `<password2>` with passwords of your choosing (remember them for the next steps). Once finished, press `Ctrl+O`, then `Enter`, and finally `Ctrl+X` to save the file and exit the editor.
```
POSTGRESQL_USERPOSTGRES_PASSWORD=<password1>
POSTGRESQL_USERMAVIRA_PASSWORD=<password2>
```
3. Run `mkdir ~/programs/postgresql` to create a directory to store our PostgreSQL passfile. Then, run `nano ~/programs/postgresql/.pgpass` to bring up an editor for our new PostgreSQL passfile. Insert the three lines below, replacing `<postgres_password>` and `<mavira_password>` with the values of `POSTGRESQL_USERPOSTGRES_PASSWORD` and `POSTGRESQL_USERMAVIRA_PASSWORD` from the `.env` file, respectively. Once finished, press `Ctrl+O`, then `Enter`, and finally `Ctrl+X` to save the file and exit the editor.
```
# hostname:port:database:username:password
localhost:5432:mavirafashiontrainingdb:postgres:<postgres_password>
localhost:5432:mavirafashiontrainingdb:mavira:<mavira_password>
```
4. Run the command `sudo passwd postgres` to create a password for the `postgres` poweruser. Enter the password for the `postgres` user that you used in step 3; i.e., the value for `POSTGRESQL_USERPOSTGRES_PASSWORD` from step 3.
5. Run `sudo find / -name initdb` to locate the location of the `initdb` command. It should be something like `/usr/lib/postgresql/<version>/bin/initdb`, where `<version>` is replaced by the major version of PostgreSQL that you installed, e.g., 17 for version 17.
6. Run `sudo su - postgres` to switch to the 'postgres' user and enter the same password you set in step 5, if prompted. Next, run `<initdb_location> -D /var/lib/postgresql/<version>/data`—replacing `<initdb_location>` with the location found in step 5 and replacing `<version>` as in step 5—to allow PostgreSQL to store data on your system at `/var/lib/postgresql/<version>/data`. Then, run `logout` to switch back from the `postgres` user to your own user.
7. Run `sudo systemctl start postgresql` to start the local PostgreSQL server, then `sudo systemctl enable postgresql` to finish setting up the server, and finally `sudo systemctl status postgresql` to verify that it is working properly.
8. Switch back to the 'postgres' user using `sudo su - postgres` and log in to psql by running `psql`.
9. From within the psql console, run `CREATE ROLE mavira WITH CREATEDB LOGIN PASSWORD '<mavira_password>';`, replacing `<mavira_password>` with the password for the `mavira` user that you used in step 3 (i.e., the value for `POSTGRESQL_USERMAVIRA_PASSWORD` from step 3) to create a new user named 'mavira'. Be sure to retain the single quotes ('') around the `<mavira_password>`.
10. Run `GRANT ALL PRIVILEGES ON DATABASE template1 TO mavira;` to enable to the `mavira` user to connect to the `template1` database.
11. Run `\q` to quit psql and then run `logout` to log out as the `postgres` user.
12. Run `psql -U mavira template1 -h localhost` to log in as the `mavira` user on the `template1` database. Enter the password that you set for the `mavira` user in step 9 when prompted.
13. Run `CREATE DATABASE mavirafashiontrainingdb;` and then `GRANT ALL PRIVILEGES ON DATABASE mavirafashiontrainingdb TO mavira;` to make sure that the `mavira` user can access the new database.
14. Run `\q` to log out of the connection to the template database. Confirm that you can connect to the new database by running `psql -U mavira mavirafashiontrainingdb -h localhost`, entering the password from step 9 again when prompted, then disconnect with `\q`.
15. Run the commands `echo 'export PGPASSFILE=~/programs/postgresql/.pgpass' >> ~/.bashrc` and `echo 'chmod 600 $PGPASSFILE' >> ~/.bashrc` to add two lines to the `~/.bashrc` file that will enable you to log in to PostgreSQL in the future as either user without needing to enter a password.


### Google Cloud Compute Engine VM Setup

#### Instance Creation
0. This section assumes that you have access to the Mavira Google Cloud Console and MaviraFashionTraining project. If you do not have access, contact  Danny if you think that you should have access.
1. From the Compute Engine dashboard of the Google Cloud Console, select Create Instance. For the purposes of this guide, the default options are fine unless otherwise noted.
2. For fashion training, the naming convention is to name the instance "fashiontraining-XXX", where XXX is a 1-indexed left-zero-padded integer denoting the number of the instance, starting with 001 and incrementing by 1 with each instance.
3. Select "us-central1 (Iowa)" as the Region and "us-central1-a" as the Zone unless desired otherwise.
4. Select the type of instance that you want to create. As a default, we recommend an n1-standard-4 instance with a T4 GPU attached, upgrading to a V100 GPU if a more powerful GPU is needed.
5. Under the OS and Storage section in the side menu, select Change under the main section and select the operating system and disk that you want. As a default, we recommend Debian GNU/Linux 12 (bookworm) x86/64, amd64; a Balanced persistent disk, and a 400 GB size. The size should be at least 40 GB for the OS and software that we need to install, plus enough storage for any datasets, model checkpoints, and other files that you need to store.
6. Select Add Local SSD towards the bottom of the same page and configure as desired. We recommend using 0 hours before timeout because important files are stored on the persistent disk. A single drive (375 GB) should be sufficient to store any datasets for active training.
7. Under Observability on the left, enable the Ops Agent option. Note the message underneath, as we will have to set this up later.
8. Under Advanced, select Spot as the Provisioning Model, as this gives significant cost savings.
9. Select Create to create the instance, then select the instance and start it once created.

#### Connect to Instance

##### Connecting for the First Time
0. We recommend using VS Code to connect to remote instances using the Google Cloud Code and Remote - SSH extensions. The rest of this guide will assume that you are using VS Code and have installed those extensions.
1. In the Google Cloud Code extension, select Compute Engine and then click the button to sign into Google Cloud, which will open in your browser. If you try to sign in using the browser and it reports that "Something went wrong", try using Google Chrome to sign in if you are not already by closing the sign-in tab, cancelling the VS Code popup about signing in, re-clicking the sign in button, and copying the address that is opened in the browser and entering it into Google Chrome. If the issue persists, contact Danny for support.
2. Return to VS Code and select a Google Cloud project (MaviraFashionTraining). When you look at the Google Cloud Code extension Compute Engine section, you should now see that the instance that you just created is running.
3. Install the Google Cloud CLI on your local machince by following this guide: https://cloud.google.com/sdk/docs/install. Then initialize it by following this guide https://cloud.google.com/sdk/docs/initializing. Optionally but recommended, select MaviraFashionTraining as the default project, and the default region and zones that the instance that you want to work on is in. The default options for everything else should be fine.
4. Enable SSH connections with your instance by following this guide: https://cloud.google.com/sdk/gcloud/reference/compute/config-ssh. You will have to specify an SSH config file to modify; any choice is fine, but make sure to remember your choice, as this file will be updated whenever the instance's external IP address changes, and you may end up having to do this manually if running into issues with the gcloud CLI (step 1 in the __Reconnecting After the First Time__ section below).
5. In the Remote Explorer extension, in the SSH tab, you should now see your instance listed. Select one of the Open options to connect to your instance. If using VS Code, select the operating system that the instance is running, if asked.
6. Open a terminal and check the prompt. It should read `<your_username>@<instance_name>:~$`.
7. If your instance has a local SSD attached, follow the instructions at the following link to format and mount it so that you can use it: https://cloud.google.com/compute/docs/disks/add-local-ssd?authuser=2#formatandmount. Be sure to continue past the first seven steps to the part that begins, "Optionally, you can add the Local SSD to the /etc/fstab file...", and complete the two remaining steps so that the local SSD is automatically formatted and mounted when you connect in future sessions.

##### Reconnecting After the First Time
1. Open a terminal in a local VS Code window and run `gcloud compute config-ssh` (See the __Other Notes__ section below if you run into difficulties). Assuming you have a running instance (If not already running, start the instance first using the Google Cloud Console or gcloud CLI), this will update the SSH config file that you chose in step 4 of the __Connecting for the First Time__ section above. If this is giving you trouble, you can get the External IP address by opening the dropdown for the desired instance in the Google Cloud Code extension or by looking at the Google Cloud Console dashboard for the instance, open the SSH config file that you chose in step 4 of the __Connecting for the First Time__ section above, and manually replace the old HostName entry for the instance's record with the new IP address.
2. In the Remote Explorer extension's SSH tab, refresh the available SSH connections and then open the instance that you want to connect to.
3. Open a terminal and check the prompt to verify that you are connected correctly as when initially connecting.

#### Starting and Stopping Instances
1. One option is to both start and stop instances from the Google Cloud Console. To do so, just navigate to the Compute Engine page, select the instance(s) that you want to start or stop, and click the start/stop button that appears at the top.
2. To start one or more instances from the gcloud CLI, simply use `gcloud compute instances start <instance_names>`, replacing `<instance_names>` with the names of the instances that you want to start. If you do not have the correct default project and/or zone set up, then add the flags `--project=<project_name>` and `--zone=<zone_name>` as needed.
3. To stop an instance, then you can run the command `sudo shutdown -h now` from within the instance. This will also run any shutdown scripts that you have set up for the instance before shutting down. If you do not have any shutdown scripts that you wish to run, you can alternatively run `gcloud compute instances stop <instance_names>` from the gcloud CLI, with the same replacements and flags mentioned in step 2 for starting an instance. Note that you may also be able to use the command `gcloud beta compute instances stop <instance_names> --discard-local-ssd=false` to store data from any local SSDs that you have attached on the persistent disk after shutdown, but this requires that the persistent disk has enough space to accommodate this additional data. For our purposes, this is likely not needed, and you may also need to remount the local SSD (see https://cloud.google.com/compute/docs/instances/stop-start-instance#restart-vm and https://cloud.google.com/compute/docs/instances/suspend-resume-instance#resume-instance)

#### Other Notes
1. A gcloud CLI cheat sheet can be found at https://cloud.google.com/sdk/docs/cheatsheet, and the full documentation at https://cloud.google.com/sdk/gcloud/reference/.
2. If you did not set up default options for the gcloud CLI when installing it or want to update your defaults, see information about how to do so at https://cloud.google.com/compute/docs/gcloud-compute#default-properties.
3. If you do not have a default project set or the default project is not the project that you need to modify, then you can always specify a project using the `--project=<project_name>` flag. Similarly, if applicable, a zone can be specified using `--zone=<zone_name>`. Other keywords can also often be specified this way.
4. For info about changing the size of persistent disks—to accommodate more training data, for instance—see https://cloud.google.com/compute/docs/disks/resize-persistent-disk.


## Reinstalling the Python Environment
To reinstall the environment from scratch to install the latest PyTorch version or install new dependencies, follow these steps:
1. If the `maviratrain` environment is not already active, run `micromamba activate maviratrain`.
2. Run `jupyter kernelspec uninstall maviratrain` and enter `y` to confirm and uninstall the associated Jupyter kernel.
3. Run `micromamba deactivate` to deactivate the environment.
4. Run `micromamba env remove --yes -n maviratrain` to remove the existing environment.
5. Run `micromamba clean --all --yes` to clear the package cache.
6. Go through the __Python Environment Setup__ section above.


## Getting Started

This section is to be completed after you have completed the __Getting Set Up__ section above. It will walk you through initializing your setup, acquiring and processing your data, and beginning to train models.

The first step is to run `bash ./scripts/initial_setup.sh` from the main directory for the FashionTraining project. This will create the directory structure for logs, checkpoints, and data initialize the tables in the `mavirafashiontrainingdb` database, run tests, and check that PyTorch recognizes any accelerators that your machine has. You can then move on to the next sections. If there are any errors or PyTorch does not recognize an accelerator that should be available, contact Danny for support.

### Acquiring and Processing Data
1. Download the dataset to your local machine from the mavira.master Google Drive. From the My Drive page, you can click the download button at the end of the aesthetic_outfit_pics folder to download it. This will download everything inside the folder as a zip file, which may take a while. You can also download each subfolder individually and place them into a directory of zip files if desired.
2. If training locally, you can move this file (or directory) into the `./data/classifier` directory (or `./data/mae` if training a Masked AutoEncoder, or MAE) of the FashionTraining project. If using a Google Cloud VM, follow the instructions at https://cloud.google.com/compute/docs/instances/transfer-files#transfergcloud to use the gcloud CLI to upload the zip file that you just downloaded from the Google Drive to the `./data/classifier` directory (or `./data/mae`) of the FashionTraining project. The `./data/classifier` and `./data/mae` directories should have been created when running the `initial_setup.sh` script in the __Getting Started__ intro. Note that changing the name of a zipped directory while it is zipped will cause problems. This is because when unzipped, the name reverts to the original and breaks the unzipping script.
3. Open up the notebook `data_processing.ipynb` found in the `./notebooks` directory of the FashionTraining project. Logs for most data processing tasks will be placed in the `./logs/data_processing` directory before being zipped and moved to the `./logs/archive/data_processing` directory once processing has terminated successfully. Important notices and warnings will also be printed to the notebook's console.
4. Follow the instructions in the first code cell to unzip the dataset into the `data` directory so that it is ready for further processing. If you are using a Google Cloud VM, make sure that `unzip` is installed first; on Debian, run `sudo apt install zip unzip` before running the first cell.
5. Follow the instructions in the second code cell to register the dataset in the PostgreSQL database.
6. Running the third code cell prints help info for the data processing script; info is also provided in the following cell.
7. Follow the instructions in the fourth code cell to run the data processing pipeline. Depending on what processing is being done and the size of the dataset, this may take up to about an hour. Once this is completed, the data is ready for training.

### Training Models
0. Currently, only classifier training has been implemented, so MAE training does not yet work. We will update this in the future when we have implemented MAE training.
1. Make a copy of the file `./scripts/run_classifier_training.py` called `./scripts/local_run_classifier_training.py`
2. Work through the file to set up PyTorch datasets and dataloaders, the model to train, optimizers and learning rate schedulers, and other hyperparameters, then train your model! Checkpoints will automatically be saved to the `./checkpoints/classifier` or `./checkpoints/mae` directories, logs will be saved to the `./logs/train_runs` directory before being zipped and stored in the `./logs/archive/train_runs` directory after training has terminated successfully, and important results and updates will also be printed to the notebook's console.


## Help
Contact Danny Collinson (dannycollinson12@gmail.com or danny@maviraai.com) if you have any questions or need support.
