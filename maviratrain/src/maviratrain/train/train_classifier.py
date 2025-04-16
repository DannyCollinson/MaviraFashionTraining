"""
Module defining training routines for Mavira's classifier models.
"""

import time
from collections.abc import Callable

import torch
from torch import GradScaler, autocast, nn, no_grad, optim
from torch.utils.data import DataLoader
from torcheval.metrics import MulticlassAccuracy
from torchvision.transforms.v2 import Transform

from maviratrain.utils.general import (
    get_device,
    get_logger,
    get_time,
    round_to_sig_figs,
)

# set up logger
logger = get_logger(
    "mt.train.train_classifier",
    log_filename=("../logs/train_runs/classifier/train_classifier.log"),
    rotation_params=(1000000, 1000),  # 1 MB, 1000 backups
)


class Trainer:
    """Trains a model given the necessary components."""

    def __init__(
        self,
        # should be (train_loader, [val_loader]), i.e., val_loader is optional
        loaders: tuple[DataLoader, DataLoader | None],
        # should be [optimizer, [optimizer(s)], [scheduler(s)]]
        # i.e., there must be at least one optimizer in the list,
        # and any number of additional optimizers or LR schedulers is okay
        optimization: list[optim.Optimizer | optim.lr_scheduler.LRScheduler],
        loss_fn: Callable,
        inloop_transforms: Transform | nn.Sequential | None = None,
        device: str = get_device(),
        automatic_mixed_precision: bool = True,
    ) -> None:
        """
        Creates a Trainer object that can train a model

        Args:
            loaders (list[DataLoader]): list of one or two DataLoaders.
                If only one, it is assumed to be the train loader.
                If two, the first is assumed to be the train loader,
                and the second will be used for validation
            optimization: (
                    list[
                        optim.Optimizer | optim.lr_scheduler.LRScheduler
                    ]
                ) list of at least one optimizer containing
                    the model to be trained's parameters.
                    Multiple optimizers may be passed if needed,
                    as well as any number of learning rate schedulers
            loss_fn (Callable): the loss function to be computed when
                comparing model output to data labels
            inloop_transforms (Transform | Sequential | None):
                transform(s) to be applied batch-wise within the
                training loop using the form
                x, y = inloop_transforms(x, y).
                Defaults to None.
            device (str): the device to train on.
                Defaults to the available device as determined by
                get_device() in maviratrain.utils.general
            automatic_mixed_precision (bool): whether or not training
                should use automatic_mixed_precision. Defaults to True.
        """
        # first loader is assumed to be train_loader
        self.train_loader = loaders[0]
        self.train_stats: list[list[float]] = [
            [],
            [],
            [],
        ]  # losses, acc@1, acc@5
        # if there is a second loader, assume it is val_loader
        if len(loaders) > 1:
            self.val_loader = loaders[1]
            self.val_stats: list[list[float]] = [
                [],
                [],
                [],
            ]  # losses, acc@1, acc@5
        else:
            self.val_loader = None

        # create list of all optimizers provided
        self.optimizers = [
            item for item in optimization if isinstance(item, optim.Optimizer)
        ]
        # create list of all learning rate schedulers provided
        self.schedulers = [
            item
            for item in optimization
            if isinstance(item, optim.lr_scheduler.LRScheduler)
        ]

        self.loss_fn = loss_fn

        self.inloop_transforms = inloop_transforms

        self.device = device

        self.automatic_mixed_precision = automatic_mixed_precision
        # override automatic mixed precision setting if running on mps
        if self.device == "mps":
            self.automatic_mixed_precision = False

        self.grad_scaler = GradScaler(
            device=self.device, enabled=self.automatic_mixed_precision
        )

    def train_one_epoch(
        self, model: nn.Module, max_steps: int | None = None
    ) -> tuple[float, float, float]:
        """
        Performs the training step for one epoch or for max_steps

        Args:
            model: (nn.Module) the PyTorch model to be trained
            max_steps: (int | None) an optional number of steps to train
                for before cutting off training.
                Only has an effect if max_steps < len(self.train_loader)

        Returns:
            float: the average loss over the number of steps trained
            float: the accuracy at k=1 during training
            float: the accuracy at k=5 during training
        """
        # make sure input is batched for compatibility
        assert (
            self.train_loader.batch_size is not None
        ), "Batch size must not be None"

        # determine the number of steps to train for
        if max_steps is not None:
            max_steps = min(len(self.train_loader), max_steps)
        else:
            max_steps = len(self.train_loader)

        model.train()  # make sure model is in training mode
        total_loss = 0  # track cumulative loss across epoch
        acc1 = MulticlassAccuracy(k=1).to(self.device)
        acc5 = MulticlassAccuracy(k=5).to(self.device)

        for step, (x, y) in enumerate(self.train_loader):
            # make sure gradient is reset for each batch for all optimizers
            for optimizer in self.optimizers:

                optimizer.zero_grad(set_to_none=True)  # default=True

            # make sure samples and labels are on correct device
            x = x.to(device=self.device)  # TODO: implement multi-GPU
            y = y.to(device=self.device)  # TODO: implement multi-GPU

            # evaluate f(x) and calculate loss
            # apply inloop transforms if needed
            if self.inloop_transforms is not None:
                x_prime, y_prime = self.inloop_transforms(x, y)
                # forward pass
                with autocast(
                    device_type=self.device,
                    enabled=self.automatic_mixed_precision,
                ):
                    fx = model(x_prime)  # forward pass
                    loss = self.loss_fn(fx, y_prime)
            else:
                with autocast(
                    device_type=self.device,
                    enabled=self.automatic_mixed_precision,
                ):
                    fx = model(x)  # forward pass
                    loss = self.loss_fn(fx, y)

            self.grad_scaler.scale(loss).backward()  # backward pass

            # model parameter updates
            for optimizer in self.optimizers:
                self.grad_scaler.step(optimizer)

            # if there are any batch-level schedulers, update after batch
            for scheduler in self.schedulers:
                if isinstance(
                    scheduler,
                    (
                        optim.lr_scheduler.OneCycleLR,
                        optim.lr_scheduler.CosineAnnealingLR,
                        optim.lr_scheduler.CyclicLR,
                        optim.lr_scheduler.ExponentialLR,
                    ),
                ):
                    scheduler.step()

            # update gradient scaler after each batch
            self.grad_scaler.update()

            total_loss += loss.detach().item()  # add batch loss to total

            # update accuracy metrics
            acc1.update(fx, y)
            acc5.update(fx, y)

            # if next step reaches max_steps, then stop training
            if step == max_steps - 1:
                break

        # average across steps (i.e., batches)
        avg_sample_loss = total_loss / max_steps
        return (
            avg_sample_loss,
            acc1.compute().item() * 100,
            acc5.compute().item() * 100,
        )

    def validation(self, model: nn.Module) -> tuple[float, float, float]:
        """
        Runs through the validation data once
        and returns the average loss

        Args:
            model: (nn.Module) the PyTorch model to evaluate

        Returns:
            float: the average loss on the validation set
            float: the top-1 accuracy on the validation set
            float: the top-5 accuracy on the validation set
        """
        # make sure the trainer has a dataset to perform evaluation on
        assert (
            self.val_loader is not None
        ), "Attempting to run validation without a validation dataset"

        # make sure input is batched for compatibility
        assert (
            self.val_loader.batch_size is not None
        ), "Batch size must not be None"

        model.eval()  # make sure model is not in training mode
        total_loss = 0  # track cumulative loss
        acc1 = MulticlassAccuracy(k=1).to(self.device)
        acc5 = MulticlassAccuracy(k=5).to(self.device)

        with no_grad():
            for x, y in self.val_loader:
                # make sure samples and labels are on correct device
                x = x.to(device=self.device)  # TODO: implement multi-GPU
                y = y.to(device=self.device)  # TODO: implement multi-GPU

                # evaluate f(x) and calculate loss
                # apply inloop transforms if needed
                # if self.inloop_transforms is not None:
                #     x_prime, y_prime = self.inloop_transforms(x, y)
                #     fx = model(x_prime)  # forward pass
                #     loss = self.loss_fn(fx, y_prime)  # provided loss fxn
                # else:
                #     fx = model(x)  # forward pass
                #     loss = self.loss_fn(fx, y)  # provided loss function

                fx = model(x)  # forward pass
                loss = self.loss_fn(fx, y)  # user-supplied loss function

                total_loss += loss.item()  # add batch loss to running loss

                # update accuracy metrics
                acc1.update(fx, y)
                acc5.update(fx, y)

        # average across batches
        avg_sample_loss = total_loss / len(self.val_loader)
        return (
            avg_sample_loss,
            acc1.compute().item() * 100,
            acc5.compute().item() * 100,
        )

    def train(
        self,
        model: nn.Module,
        n_epochs: int | None = None,
        n_steps: int | None = None,
        val_interval: int = 1,
        freeze_epochs: int = 0,
        stopping_intervals: tuple[int | None, int | None] = (None, None),
    ) -> tuple[nn.Module, int, int]:
        """
        Trains the model for n_epochs or n_steps

        Args:
            model: (nn.Module) the PyTorch model to train
            n_epochs (int | None): number of epochs to train. None if
                training based on steps. If n_epochs is not None,
                then n_epochs must be None. Defaults to None
            n_steps (int | None): number of steps to train for
                If n_steps is not None, n_epochs must be None.
                Defaults to None
            val_interval (int): number of epochs between validation runs
            freeze_epochs (int): number of epochs for which to freeze
                the backbone of the model. Defaults to 0 (no freezing).
            stopping_intervals (tuple[int | None, int | None]): number
                of train/validation epochs/evaluations without
                improvement in loss or accuracy after which to stop
                training early.
                Defaults to (None, None) (no early stopping).

        Returns:
            nn.Module: the trained model
            int: number of steps trained for
            int: number of epochs trained for
        """
        # make sure the time to train for is provided
        assert (
            n_epochs is not None or n_steps is not None
        ), "Either n_epochs or n_steps must not be None"

        # make sure the time to train for is unambiguous
        assert (
            n_epochs is None or n_steps is None
        ), "Specify either n_epochs or n_steps, but not both"

        # make sure the stopping interval numbers are valid
        assert all(num is None or num >= 0 for num in stopping_intervals), (
            "Stopping intervals must be None or integer >= 0, "
            f"got {stopping_intervals}"
        )

        # if n_epochs is provided, find the number of steps needed
        if n_steps is None:
            # this line is unnecessary; it's only here to appease Pylance
            n_epochs = n_epochs if n_epochs is not None else 0
            n_steps = n_epochs * len(self.train_loader)

        # track the number of steps to go before stopping training
        steps_left = n_steps
        cur_epoch = -1  # track the current epoch
        start_time = time.time()  # track the total training time

        # track best val stats for model checkpointing and early stopping
        best_val_loss = torch.inf
        best_val_acc1 = 0.0
        best_val_acc5 = 0.0

        # track best train stats for early stopping
        best_train_loss = torch.inf
        best_train_acc1 = 0.0
        best_train_acc5 = 0.0

        # track evaluations without improvement for early stopping
        train_no_improvement = 0
        val_no_improvement = 0

        try:  # except KeyboardInterrupt for stopping training
            while steps_left > 0:
                # if available and scheduled, run validation
                # note: validation is run before training to get a baseline
                # for metrics and training time before beginning real training
                if (
                    self.val_loader is not None
                    and (cur_epoch + 1) % val_interval == 0
                ):
                    val_start_time = time.time()
                    val_loss, val_acc1, val_acc5 = self.validation(model=model)
                    self.val_stats[0].append(val_loss)  # record val loss
                    self.val_stats[1].append(val_acc1)  # record val accuracy
                    self.val_stats[2].append(val_acc5)  # record val acc@5
                    val_end_time = time.time()
                    logger.info_(
                        f"Val Epoch: {cur_epoch + 1}        "  # 8 spaces
                        f"Loss: {val_loss:.4f}        "  # 8 spaces
                        f"Accuracy: {val_acc1:.2f}        "  # 8 spaces
                        f"Top-5 Accuracy: {val_acc5:.2f}        "  # 8 spaces
                        f"Time: {val_end_time - val_start_time:.2f}"
                    )

                    # save model checkpoint if a validation improvement is made
                    if cur_epoch >= 0 and (
                        val_loss < best_val_loss
                        or val_acc1 > best_val_acc1
                        or val_acc5 > best_val_acc5
                    ):
                        val_no_improvement = 0
                        best_val_acc1 = max(val_acc1, best_val_acc1)
                        best_val_acc5 = max(val_acc5, best_val_acc5)
                        best_val_loss = min(val_loss, best_val_loss)
                        model_path = (
                            "../checkpoints/classifier/models/"
                            f"jX"
                            f"-A{"0" * (2 - int(best_val_acc1))}"
                            f"{best_val_acc1:.1f}"
                            f"-a{"0" * (2 - int(best_val_acc5))}"
                            f"{best_val_acc5:.1f}"
                            f"-E{(cur_epoch + 1):03}"
                            f"-T{get_time().replace(":", "-")}.pt"
                        )
                        torch.save(model.state_dict(), model_path)

                        # save optimizer states
                        optimizer_general_path = (
                            model_path.replace("models", "optimizers") + "-oX"
                        )
                        for ind, optimizer in enumerate(self.optimizers):
                            optimizer_path = optimizer_general_path[:-1] + str(
                                ind
                            )
                            torch.save(optimizer.state_dict(), optimizer_path)

                        # save scheduler states
                        scheduler_general_path = (
                            model_path.replace("models", "schedulers") + "-sX"
                        )
                        for ind, scheduler in enumerate(self.schedulers):
                            scheduler_path = scheduler_general_path[:-1] + str(
                                ind
                            )
                            torch.save(scheduler.state_dict(), scheduler_path)

                        logger.info(
                            "Saved model checkpoint to %s, "
                            "optimizer state(s) to %s, "
                            "and scheduler state(s) to %s",
                            model_path,
                            optimizer_general_path,
                            scheduler_general_path,
                        )
                    else:
                        val_no_improvement += 1

                        # if enough validation runs without improvement
                        if val_no_improvement == stopping_intervals[1]:
                            raise InterruptedError(
                                f"{val_no_improvement} validation runs "
                                "without improvement"
                            )

                train_start_time = time.time()  # track the time per epoch

                cur_epoch += 1  # update epoch count

                # unfreeze backbone if reached freeze_epochs epochs
                if cur_epoch == freeze_epochs:
                    for params in model.parameters():
                        params.requires_grad_(True)

                # determine number of steps to train for and perform training
                epoch_steps = min(steps_left, len(self.train_loader))
                train_loss, train_acc1, train_acc5 = self.train_one_epoch(
                    model=model, max_steps=epoch_steps
                )
                self.train_stats[0].append(train_loss)  # record training loss
                self.train_stats[1].append(train_acc1)  # record training acc
                self.train_stats[2].append(train_acc5)  # record training acc@5

                # update stats and check for early stopping
                train_no_improvement += 1
                if train_loss < best_train_loss:
                    best_train_loss = train_loss
                    train_no_improvement = 0
                if train_acc1 > best_train_acc1:
                    best_train_acc1 = train_acc1
                    train_no_improvement = 0
                if train_acc5 > best_train_acc5:
                    best_train_acc5 = train_acc5
                    train_no_improvement = 0
                if train_no_improvement == stopping_intervals[0]:
                    raise InterruptedError(
                        f"{train_no_improvement} training epochs "
                        "without improvement"
                    )

                steps_left -= epoch_steps  # adjust remaining steps

                # if there are any epoch-level schedulers, update after epoch
                for scheduler in self.schedulers:
                    if not isinstance(
                        scheduler,
                        (
                            optim.lr_scheduler.OneCycleLR,
                            optim.lr_scheduler.CosineAnnealingLR,
                            optim.lr_scheduler.CyclicLR,
                            optim.lr_scheduler.ExponentialLR,
                        ),
                    ):
                        scheduler.step()

                # get learning rates for display
                learning_rates = [
                    [
                        round_to_sig_figs(param_group["lr"], 3)
                        for param_group in optimizer.param_groups
                    ]
                    for optimizer in self.optimizers
                ]
                # extract from lists if possible for display
                if len(learning_rates) == 1:
                    learning_rates = learning_rates[0]  # type: ignore
                    if len(learning_rates) == 1:
                        learning_rates = learning_rates[0]  # type: ignore

                train_end_time = time.time()
                logger.info_(
                    f"Train Epoch: {cur_epoch + 1}        "  # 8 spaces
                    f"Loss: {train_loss:.4f}        "  # 8 spaces
                    f"Accuracy: {train_acc1:.2f}        "  # 8 spaces
                    f"Top-5 Accuracy: {train_acc5:.2f}        "  # 8 spaces
                    f"Learning Rate(s): {learning_rates}        "  # 8
                    f"Time: {train_end_time - train_start_time:.2f}"
                )

            # run one more validation
            if self.val_loader is not None:  # only run if we have val data
                logger.info_(
                    "Running final validation after "
                    f"{cur_epoch + 1} training epochs:"
                )
                val_loss, val_acc1, val_acc5 = self.validation(model=model)
                self.val_stats[0].append(val_loss)  # record val loss
                self.val_stats[1].append(val_acc1)  # record val accuracy
                self.val_stats[2].append(val_acc5)  # record val acc@5
                logger.info_(
                    f"Loss: {val_loss:.4f}        "  # 8 spaces
                    f"Accuracy: {val_acc1:.2f}        "  # 8 spaces
                    f"Top-5 Accuracy: {val_acc5:.2f}        "  # 8 spaces
                    f"Total Time: {time.time() - start_time:.2f}"
                )

                # save model checkpoint if a validation improvement is made
                if cur_epoch >= 0 and (
                    val_loss < best_val_loss
                    or val_acc1 > best_val_acc1
                    or val_acc5 > best_val_acc5
                ):
                    best_val_acc1 = max(val_acc1, best_val_acc1)
                    best_val_acc5 = max(val_acc5, best_val_acc5)
                    best_val_loss = min(val_loss, best_val_loss)
                    model_path = (
                        "../checkpoints/classifier/models/"
                        f"jX"
                        f"-A{"0" * (2 - int(best_val_acc1))}"
                        f"{best_val_acc1:.1f}"
                        f"-a{"0" * (2 - int(best_val_acc5))}"
                        f"{best_val_acc5:.1f}"
                        f"-E{(cur_epoch + 1):03}"
                        f"-T{get_time().replace(":", "-")}.pt"
                    )
                    torch.save(model.state_dict(), model_path)

                    # save optimizer states
                    optimizer_general_path = (
                        model_path.replace("models", "optimizers") + "-oX"
                    )
                    for ind, optimizer in enumerate(self.optimizers):
                        optimizer_path = optimizer_general_path[:-1] + str(ind)
                        torch.save(optimizer.state_dict(), optimizer_path)

                    # save scheduler states
                    scheduler_general_path = (
                        model_path.replace("models", "schedulers") + "-sX"
                    )
                    for ind, scheduler in enumerate(self.schedulers):
                        scheduler_path = scheduler_general_path[:-1] + str(ind)
                        torch.save(scheduler.state_dict(), scheduler_path)

                    # save grad_scaler state if applicable
                    if self.automatic_mixed_precision:
                        scaler_path = (
                            model_path.replace("models", "scalers") + "-gX"
                        )
                        torch.save(self.grad_scaler.state_dict(), scaler_path)

                    logger.info(
                        "Saved model checkpoint to %s, "
                        "optimizer state(s) to %s, "
                        "and scheduler state(s) to %s",
                        model_path,
                        optimizer_general_path,
                        scheduler_general_path,
                    )

        except (KeyboardInterrupt, InterruptedError) as error:
            if len(error.args) == 0:
                reason = "keyboard interrupt"
            else:
                reason = error.args[0]
            logger.info_("Training stopped by %s", reason)

        return model, n_steps - steps_left, cur_epoch
