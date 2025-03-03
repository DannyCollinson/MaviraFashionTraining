"""
Module defining the dataset class for Mavira's Classifier training.
"""

from pathlib import Path
from typing import Any

from torch import Tensor, ones_like, tensor  # pylint: disable=E0611
from torch.nn import Sequential
from torch.utils.data import DataLoader
from torchvision.tv_tensors import Image
from torchvision.datasets import DatasetFolder
from torchvision.transforms.v2 import Transform

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
        transforms: Sequential | Transform | None = None,
        loss_weights_balancing_factor: float = 0.0,
    ) -> None:
        """
        Initializes dataset by specifying the path to the data and
        any transforms to apply to the images

        Args:
            data_path (Path | str): path to the image data
            transforms (Sequential | Transform | None):
                transforms to apply to the images
            loss_weights_balancing_factor (float): value from 0 to 1 (*)
                that adjusts the strength of loss weighting for class
                balance. 0 means no balancing and weights are all 1.0.
                1 means full balancing where weights are set to give
                each class equal influence. Values between 0 and 1
                specify a linear combination of uniform and balanced
                weights; e.g., a value of 0.6 uses the formula
                weights = 0.6 * balanced + (1 - 0.6) * uniform
                (*): value can technically be outside the range [0, 1],
                but values less than 0 create anti-balancing,
                and values greater than 1 create over-balancing
        """
        # check that loss_weights_balancing_factor is in the range [0, 1]
        assert (
            loss_weights_balancing_factor >= 0
            and loss_weights_balancing_factor <= 1
        ), (
            "Value for loss_weights_balancing_factor must be in the range "
            f"[0, 1], got {loss_weights_balancing_factor}"
        )

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

        self.device = get_device(verbose=False)

        # self.classes is list[str] of class names
        # self.class_to_index = dict with class:str keys and index:int values

        # self.samples is a list of sample paths and class indices
        # i.e., [(sample_path, class_index)]

        # calculate the number of train samples per label for loss weighting
        self.train_samples_per_label: dict[int, int] = {}
        for _, label in self.samples:
            if label not in self.train_samples_per_label:
                self.train_samples_per_label[label] = 0
            self.train_samples_per_label[label] += 1
        self.class_counts = tensor(
            [self.train_samples_per_label[label] for _, label in self.samples]
        )

        # compute the loss weights
        # len(samples) / len(classes) = % of total loss allocated to each class
        #     if uniformly distributed amongst classes
        # divide by class_counts -> each sample from the class splits
        #     the class's percentage of the loss equally
        self.standard_loss_weights = len(self.samples) / (
            len(self.classes) * self.class_counts
        )

        # adjust loss weighting by scaling factor
        self.loss_weights = (
            loss_weights_balancing_factor * self.standard_loss_weights
            + (1 - loss_weights_balancing_factor)
            * ones_like(self.class_counts)
        )

        self.transforms = transforms

    def __len__(self) -> int:
        """
        Returns the number of items in the dataset

        Returns:
            int: number of items in the dataset
        """
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Image, int]:
        """
        Returns the image and corresponding class index that
        are at the specified index in the samples list

        Args:
            index (int): index of the dataset item to retrieve

        Returns:
            tuple[Image, int]: the image and the class label index
        """
        img_path, class_index = self.samples[index]

        img = self.loading_function(img_path)
        img = Image(img)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, class_index


def make_training_dataloaders(
    train_data_path: Path | str,
    transforms: Sequential | Transform | None = None,
    train_dataloader_params: dict[str, Any] | None = None,
    val_data_path: Path | str | None = None,
    val_dataloader_params: dict[str, Any] | None = None,
    loss_weights_balancing_factor: float = 0.0,
) -> tuple[
    ClassifierDataset, DataLoader, ClassifierDataset | None, DataLoader | None
]:
    """
    Create training and validation datasets and dataloaders
    for Mavira's Classifier training

    Args:
        train_data_path (Path | str): path to the training data
        transforms (Sequential | Transform | None):
            transforms to apply to the images
        train_dataloader_params (dict[str, Any] | None): parameters for
            creating the training dataloader. Defaults to None.
            Specified parameters will replace the default values.
        val_data_path (Path | str | None): path to the validation data.
            If None, the validation dataset will not be created.
            Defaults to None.
        val_dataloader_params (dict[str, Any] | None): parameters for
            creating the validation dataset. Defaults to None.
            Specified parameters will replace the default values.
        loss_weights_balancing_factor (float): the balancing
            factor for loss weighting

    Returns:
        tuple[
            ClassifierDataset,
            DataLoader,
            ClassifierDataset | None,
            DataLoader | None
        ]: the training dataset, training dataloader,
            validation dataset, and validation dataloader
    """
    # create training dataset, including transforms if specified

    train_dataset = ClassifierDataset(
        data_path=train_data_path,
        transforms=transforms,
        loss_weights_balancing_factor=loss_weights_balancing_factor,
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

    # create validation dataset, including transforms if specified
    val_dataset = ClassifierDataset(
        data_path=val_data_path,
        transforms=transforms,
        loss_weights_balancing_factor=loss_weights_balancing_factor,
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
