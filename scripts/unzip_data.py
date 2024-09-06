"""Script to unzip downloaded images into data folder"""

import os

os.system(
    "cd /mnt/c/Users/danny/Downloads && "
    "unzip *_files-*.zip "
    "-d /home/danny/mavira/FashionTraining/data/classification44"
)
