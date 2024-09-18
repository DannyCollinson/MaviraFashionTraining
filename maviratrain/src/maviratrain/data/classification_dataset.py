"""File defining the dataset for Mavira's Classifier training"""

from pathlib import Path
from typing import Union

import torch
from torch import Tensor
from torchvision.datasets import DatasetFolder
from torchvision.transforms.v2 import RandomHorizontalFlip

from ..utils.general import get_device, is_valid_directory


class ClassifierDataset(DatasetFolder):
    """
    PyTorch Dataset for Mavira's Classifier training

    Subclasses torchvision's DatasetFolder class
    """

    def __init__(self, data_path: Union[Path, str]) -> None:
        """
        Initializes dataset by specifying the path to the data

        Arguments:
            data_path {Union[Path, str]} -- path to the image data
        """
        # super is DatasetFolder, which inherits from VisionDataset
        super().__init__(
            root=data_path,
            loader=torch.load,
            extensions=(".pt",),
        )

        is_valid_directory(data_path=data_path)  # check data directory

        # set object's attribute as pathlib Path
        if isinstance(data_path, str):
            self.data_path = Path(data_path)
        elif isinstance(data_path, Path):
            self.data_path = data_path

        # self.classes is list[str] of class names
        # self.class_to_index = dict with class:str keys and index:int values

        # self.samples is a list of sample paths and class indices
        # i.e., [(sample_path, class_index)]

        self.device = get_device(verbose=False)

        self.transform = RandomHorizontalFlip()

    def __len__(self) -> int:
        """
        Returns the number of items in the dataset

        Returns:
            int -- number of items in the dataset
        """
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        """
        Returns the image and corresponding class index that
        are at the specified index in the samples list

        Arguments:
            index {int} -- index of the dataset item to retrieve

        Returns:
            tuple[Tensor, int] -- the image and the class label index
        """
        img_path, class_index = self.samples[index]

        img = torch.load(img_path, map_location=self.device, weights_only=True)
        img = self.transform(img)

        return img, class_index
