"""
Module defining the dataset class for Mavira's Classifier training.
"""

from pathlib import Path
from typing import Any

from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder
from torchvision.transforms.v2 import RandomHorizontalFlip, Transform

from ..utils.constants import (
    DEFAULT_TRAIN_DATALOADER_PARAMS,
    DEFAULT_VAL_TEST_DATALOADER_PARAMS,
)
from ..utils.general import (
    get_dataset_extension,
    get_device,
    get_loading_function,
    is_valid_dataset,
)


class ClassifierDataset(DatasetFolder):
    """
    PyTorch Dataset for Mavira's Classifier training

    Subclasses torchvision's DatasetFolder class
    """

    def __init__(
        self,
        data_path: Path | str,
        additional_transforms: Transform | None = None,
    ) -> None:
        """
        Initializes dataset by specifying the path to the data and
        any additional transforms to apply to the images

        Args:
            data_path (Path | str): path to the image data
            additional_transforms (Transform | None): additional
                transforms to apply to the images
        """
        # get the extension of the dataset to determine the loading function
        self.extension = get_dataset_extension(data_path=data_path)
        self.loading_function = get_loading_function(extension=self.extension)

        # super is DatasetFolder, which inherits from VisionDataset
        super().__init__(
            root=data_path,
            loader=self.loading_function,
            extensions=(self.extension,),
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

        self.transform = RandomHorizontalFlip()
        self.additional_transforms = additional_transforms

    def __len__(self) -> int:
        """
        Returns the number of items in the dataset

        Returns:
            int: number of items in the dataset
        """
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        """
        Returns the image and corresponding class index that
        are at the specified index in the samples list

        Args:
            index (int): index of the dataset item to retrieve

        Returns:
            tuple[Tensor, int]: the image and the class label index
        """
        img_path, class_index = self.samples[index]

        img = self.loading_function(img_path)
        img = self.transform(img)
        if self.additional_transforms is not None:
            img = self.additional_transforms(img)

        return img, class_index


def make_training_dataloaders(
    train_data_path: Path | str,
    additional_transforms: Transform | None = None,
    train_dataloader_params: dict[str, Any] | None = None,
    val_data_path: Path | str | None = None,
    val_dataloader_params: dict[str, Any] | None = None,
) -> tuple[
    ClassifierDataset, DataLoader, ClassifierDataset | None, DataLoader | None
]:
    """
    Create training and validation datasets and dataloaders
    for Mavira's Classifier training

    Args:
        train_data_path (Path | str): path to the training data
        additional_transforms (Transform | None): additional
            transforms to apply to the images
        train_dataset_params (dict[str, Any] | None): parameters for
            creating the training dataloader. Defaults to None.
            Specified parameters will replace the default values.
        val_data_path (Path | str | None): path to the validation data.
            If None, the validation dataset will not be created.
            Defaults to None.
        val_dataset_params (dict[str, Any] | None): parameters for
            creating the validation dataset. Defaults to None.
            Specified parameters will replace the default values.

    Returns:
        tuple[
            ClassifierDataset,
            DataLoader,
            ClassifierDataset | None,
            DataLoader | None
        ]: the training dataset, training dataloader,
            validation dataset, and validation dataloader
    """
    # create training dataset, including additional transforms if specified
    if additional_transforms is None:
        train_dataset = ClassifierDataset(data_path=train_data_path)
    else:
        train_dataset = ClassifierDataset(
            data_path=train_data_path,
            additional_transforms=additional_transforms,
        )

    # create training dataloader with specified parameters
    # default parameters are replaced by any specified in the function call
    train_dataloader_parameters = {**DEFAULT_TRAIN_DATALOADER_PARAMS}
    if train_dataloader_params is not None:
        train_dataloader_parameters.update(train_dataloader_params)
    train_dataloader: DataLoader = DataLoader(
        dataset=train_dataset, **train_dataloader_parameters  # type: ignore
    )

    # if no validation data path is specified, return the training dataset
    if val_data_path is None:
        return train_dataset, train_dataloader, None, None

    # create validation dataset, including additional transforms if specified
    if additional_transforms is None:
        val_dataset = ClassifierDataset(data_path=val_data_path)
    else:
        val_dataset = ClassifierDataset(
            data_path=val_data_path,
            additional_transforms=additional_transforms,
        )

    # create validation dataloader with specified parameters
    # default parameters are replaced by any specified in the function call
    val_dataloader_parameters = {**DEFAULT_VAL_TEST_DATALOADER_PARAMS}
    if val_dataloader_params is not None:
        val_dataloader_parameters.update(val_dataloader_params)
    val_dataloader: DataLoader = DataLoader(
        dataset=val_dataset, **val_dataloader_parameters  # type: ignore
    )

    return train_dataset, train_dataloader, val_dataset, val_dataloader
