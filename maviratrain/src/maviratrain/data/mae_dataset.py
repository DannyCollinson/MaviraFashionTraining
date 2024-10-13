""" Module defining the dataset class for Mavira's MAE training. """

from pathlib import Path

import torch
from torch import Tensor
from torchvision.datasets import DatasetFolder
from torchvision.transforms.v2 import RandomHorizontalFlip

from ..utils.general import get_device, is_valid_dataset


class MAEDataset(DatasetFolder):
    """
    PyTorch Dataset for Mavira's MAE training

    Subclasses PyTorch's Dataset class
    """

    def __init__(self, data_path: Path | str) -> None:
        """
        Initializes dataset by specifying the path to the data

        Arguments:
            data_path {Path | str} -- path to the image data
        """
        # super is DatasetFolder, which inherits from VisionDataset
        super().__init__(
            root=data_path,
            loader=torch.load,
            extensions=(".pt",),
            transform=RandomHorizontalFlip,
        )

        assert is_valid_dataset(data_path=data_path)  # check dataset directory

        # set object's attribute as pathlib Path
        if isinstance(data_path, str):
            self.data_path = Path(data_path)
        else:
            self.data_path = data_path

        # self.classes is list[str] of class names
        # self.class_to_index = dict with class:str keys and index:int values

        # self.samples is a list of sample paths and class indices
        # i.e., [(sample_path, class_index)]

        self.device = get_device(verbose=False)

    def __len__(self) -> int:
        """
        Returns the number of items in the dataset

        Returns:
            int -- number of items in the dataset
        """
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """
        Returns a mask for the image and the image from the specified index

        Arguments:
            index {int} -- index of the dataset item to retrieve

        Returns:
            tuple[Tensor, Tensor] -- the mask and the image
        """
        img_path, _ = self.samples[index]

        img = torch.load(img_path, map_location=self.device, weights_only=True)
        img = RandomHorizontalFlip()(img)

        mask = Tensor()

        return mask, img
