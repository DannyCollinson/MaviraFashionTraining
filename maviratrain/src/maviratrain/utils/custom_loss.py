"""Custom PyTorch loss functions."""

from torch import Tensor, arange  # pylint: disable=no-name-in-module
from torch.nn import CrossEntropyLoss, Module, Softmax


class MultiClassFocalLoss(Module):
    """
    Multi-class focal loss implementation inspired by "Focal Loss for
    Dense Object Detection" (https://arxiv.org/abs/1708.02002) and
    addapted to support classification tasks with more than two classes

    Note that the alpha parameter's meaning differs somewhat from its
    meaning in Lin et al.'s original binary focal loss

    This implementation also supports the `ignore_index` and
    `label_smoothing` arguments from PyTorch's `CrossEntropyLoss` class

    Note that one difference from `CrossEntropyLoss` is that if all
    samples have target value `ignore_index`, then `MultiClassFocalLoss`
    returns 0 where `CrossEntropyLoss` would return `nan`.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Tensor | None = None,
        reduction: str = "mean",
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        weight: Tensor | None = None,
    ) -> None:
        """
        Initialize multi-class focal loss function

        Args:
            gamma (float, optional): focusing parameter that determines
                importance of hard samples vs easy samples. If set to
                `0`, focal loss is identical to `CrossEntropyLoss`.
                As `gamma` grows above 0, focusing strength increases
                exponentially with `gamma`. Defaults to `2`.
            alpha (Tensor | None, optional): tensor of class balancing
                factors of shape `[num_classes,]`. Identical to the
                `weight` argument of `CrossEntropyLoss`. Should be on
                correct device before training. Defaults to `None`.
            reduction (str): identical to the `reduction` argument to
                `CrossEntropyLoss`: "Specifies the reduction to apply to
                the output". Defaults to `"mean"`.
                Values `"mean"`, `"sum"`, and `"none"` are valid.
            ignore_index (int): identical to the `ignore_index` argument
                to `CrossEntropyLoss`: "Specifies a target value that is
                ignored and does not contribute to the input gradient.
                When `reduction` is `"mean"`, the loss is averaged over
                non-ignored targets. Note that `ignore_index` is only
                applicable when the target contains class indices."
                Defaults to `-100`.
            label_smoothing (float): identical to the `label_smoothing`
                argument to `CrossEntropyLoss`: "A float in [0.0, 1.0].
                Specifies the amount of smoothing when computing the
                loss, where 0.0 means no smoothing. The targets become a
                mixture of the original ground truth and a uniform
                distribution as described in 'Rethinking the Inception
                Architecture for Computer Vision'." Defaults to 0.0.
            weight (Tensor | None, optional): alternate name for
                specifying `alpha`. Included for drop-in compatibility
                with `CrossEntropyLoss`. Ignored if `alpha` is not
                `None`. Defaults to `None`.
        """
        super().__init__()  # type: ignore

        # check reduction mode
        assert reduction in [
            "none",
            "sum",
            "mean",
        ], (
            "Valid reduction modes are 'none', 'sum', and 'mean', "
            f"got {reduction}"
        )

        # make sure gamma is numeric
        assert isinstance(
            gamma, (int, float)
        ), f"Gamma must be a float, got {gamma} of type {type(gamma)}"

        # check alpha
        if alpha is not None:
            assert isinstance(alpha, Tensor), (
                "Alpha must be a tensor or None, "
                f"got {alpha} of type {type(alpha)}"
            )
        elif weight is not None:
            assert isinstance(weight, Tensor), (
                "Weight/alpha must be a tensor or None, "
                f"got {weight} of type {type(weight)}"
            )
            # use weight in place of alpha for drop-in compatability
            alpha = weight

        # components
        self.cross_entropy = CrossEntropyLoss(
            reduction="none",
            weight=alpha,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
        )
        self.softmax = Softmax(dim=1)

        # focusing parameters
        self.gamma = float(gamma)
        self.alpha = alpha

        # other parameters
        self.label_smoothing = label_smoothing

        # settings
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs: Tensor, target: Tensor) -> Tensor:
        """
        Calculate multi-class focal loss

        Args:
            inputs (Tensor): (unnormalized) prediction logits
                of shape `[batch_size, num_classes]`
            target (Tensor): true labels of shape `[batch_size,]`

        Returns:
            Tensor: loss tensor with reduction applied
        """
        # calculate regular cross entropy loss
        cross_entropy_loss = self.cross_entropy(inputs, target)

        # calculate predicted class probabilities for correct classes
        probabilities = self.softmax(inputs)[arange(target.shape[0]), target]

        # calculate focusing if needed
        if self.gamma != 0:
            focus = (1 - probabilities) ** self.gamma
        else:
            focus = 1  # dummy

        # apply focusing
        loss = focus * cross_entropy_loss

        # apply reduction option to loss and return
        if self.reduction == "mean":
            # CrossEntropyLoss "mean" divides by effective number of samples,
            # including checks for ignored index
            if self.alpha is not None:
                weighted_sample_num = Tensor(
                    [
                        self.alpha[val] if val != self.ignore_index else 0
                        for val in target
                    ]
                ).sum()
            else:
                weighted_sample_num = Tensor(
                    [1 for val in target if val != self.ignore_index]
                ).sum()
            if weighted_sample_num.item() == 0:
                # if all targets ignored, we want to return 0, not nan
                return Tensor([0.0])
            return loss.sum() / weighted_sample_num
        if self.reduction == "sum":
            return loss.sum()
        return loss
