# Important
This is a copy of the Mavira FashionTraining repository written solely by Danny Collinson for the purposes of providing a personal code sample. No permissions are granted to anyone outside of Mavira except for those outlined in GitHub's Terms of Use.

This repository is a work in progress and will be updated periodically as changes to the parent repository are made.

# Training your Personal Fashion Stylist

The code here is for training the models used for the AI fashion assistant. It includes the `maviratrain` package.

## Getting Started
### Short Version
With the project repository cloned from GitHub (https://github.com/MaviraAI/FashionTraining.git) and `micromamba` installed, run the following command from the project directory. Then install the following VSCode Extensions as well as any others that you prefer to use and that do not conflict with the following: Python, Pylance, Pylint, Jupyter, Black Formatter, autoDocstring, isort, Mypy Type Checker.
```
micromamba create --yes -n maviratrain \
        pytorch torchvision pytorch-cuda=12.4 seaborn ipykernel sympy=1.13.1 pillow=10.4 psycopg2 python-dotenv\
        -c pytorch-nightly -c nvidia -c conda-forge --channel-priority flexible \
    && micromamba clean --all --yes \
    && micromamba activate maviratrain \
    && python -m ipykernel install --user --name maviratrain \
    && pip install fsspec pytest \
    && python -m pip install -e maviratrain
```

### Long Version
1. Make sure that `micromamba` is installed. If not, follow https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html to install.
2. Make sure that git is installed. If not, follow https://git-scm.com/downloads to install.
3. Clone this repository by running `git clone https://github.com/MaviraAI/FashionTraining.git` from your directory for Mavira projects.
4. Run the following command from the project directory and confirm by pressing `Enter` when prompted to install necessary dependencies via `micromamba`.
```
micromamba create -n maviratrain \
    pytorch torchvision pytorch-cuda=12.4 seaborn ipykernel sympy=1.13.1 pillow=10.4 psycopg2 python-dotenv\
    -c pytorch-nightly -c nvidia -c conda-forge --channel-priority flexible
```
5. Run `micromamba clean --all --yes` to clean up the `micromamba` installation.
6. Run `micromamba activate maviratrain` to activate the environment.
7. Run `python -m ipykernel install --user --name maviratrain` to create an IPython kernel for use in notebooks.
6. Run `pip install fsspec pytest`. The `micromamba` install misses `fsspec`, and we need `pytest` for testing.
8. Run `python -m pip install -e <path/to/FashionTraining>/maviratrain` to install an editable version of our training package that updates as you make changes.
9. Run `micromamba env export --no-build` and make sure that it looks like the `env.yml` file in the main project directory.
10. Install the following VSCode extensions: Python, Pylance, Pylint, Jupyter, Black Formatter, MyPy, autoDocstring, isort, Mypy Type Checker.
11. Install other VSCode extensions that you prefer to use and that do not conflict with those already installed.

## Reinstalling Environment
To reinstall the environment from scratch to install the latest PyTorch version or install new dependencies, follow these steps:
1. If the `maviratrain` environment is not already active, run `micromamba activate maviratrain`.
2. Run `jupyter kernelspec uninstall maviratrain` and enter `y` to confirm and uninstall the associated Jupyter kernel.
3. Run `micromamba deactivate` to deactivate the environment.
4. Run `micromamba env remove --yes -n maviratrain` to remove the existing environment.
5. Run the command from the Short Version of the Getting Started section above or follow steps 4-9 in the Long Version section.

## Help
Ask Danny (dannycollinson12@gmail.com or danny@maviraai.com) if you have any questions.
For now, we are all running everything locally until we move things to the cloud.
