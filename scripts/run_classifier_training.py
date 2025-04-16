"""Script to run classifier training."""

# Script setup (add imports if needed for your configuration)
###############################################################################
###############################################################################

from typing import Any

# pylint: disable=unused-import
from torch import Tensor, float32
from torch import load as torch_load
from torch import manual_seed, uint8
from torch.nn import CrossEntropyLoss, Linear, Sequential, init
from torch.optim import AdamW, lr_scheduler
from torch_focalloss import MultiClassFocalLoss
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    EfficientNet_V2_S_Weights,
    MaxVit_T_Weights,
    MobileNet_V3_Small_Weights,
    ShuffleNet_V2_X0_5_Weights,
    Swin_T_Weights,
    ViT_B_32_Weights,
    convnext_tiny,
    efficientnet_v2_s,
    maxvit_t,
    mobilenet_v3_small,
    shufflenet_v2_x0_5,
    swin_t,
    vit_b_32,
)
from torchvision.transforms.v2 import (
    ColorJitter,
    CutMix,
    MixUp,
    RandomApply,
    RandomChoice,
    RandomHorizontalFlip,
    RandomResizedCrop,
    RandomRotation,
    ToDtype,
)

from maviratrain.data.classification_dataset import make_training_dataloaders
from maviratrain.train.train_classifier import Trainer
from maviratrain.utils.constants import DEFAULT_SEED
from maviratrain.utils.custom_transforms import ImageNetNormalize
from maviratrain.utils.general import get_device, get_logger

# set up logger
logger = get_logger(
    "scripts.run_classifier_training",
    log_filename=("../logs/train_runs/classifier/run_classifier_training.log"),
    rotation_params=(1000000, 1000),  # 1 MB, 1000 backups
)

logger.info_("Beginning setup for training...")


# Configuration (edit settings below)
###############################################################################
###############################################################################

# General configuration
###############################################################################

# Set seed for reproducability ("default" for default seed of 42)
seed: int | str = "default"

# Select device to run training on (use "default" for best available device)
device = "default"

# Whether or not to train using automatic mixed precision
automatic_mixed_precision = True

# Data Configuration
###############################################################################

# The path to the dataset used for training
train_path = (
    "/home/danny/mavira/FashionTraining/data/classifier/"
    "classifier361-test-r002-s003/train/"
)

# The path to the dataset used for validation
val_path = (
    "/home/danny/mavira/FashionTraining/data/classifier/"
    "classifier361-test-r002-s003/val/"
)

# Specify any non-default dataloader parameters
# Defaults found in maviratrain.utils.constants
dataloader_params = {"batch_size": 64, "num_workers": 0}

# Specify any transforms to apply to the data
train_transforms = Sequential(
    ToDtype(uint8, scale=True),
    RandomApply([RandomResizedCrop(size=256, scale=(0.25, 1.0))], p=0.7),
    RandomHorizontalFlip(p=0.5),
    RandomApply(
        [ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)], p=0.5
    ),
    RandomApply([RandomRotation(degrees=(-10, 10), expand=False)], p=0.5),
    ImageNetNormalize(),
)
val_transforms = ImageNetNormalize()

# Training Configuration
###############################################################################

# Train time configuration

# Number of epochs for warmup and total training
warmup_epochs = 10
total_epochs = 200

# Number of epochs to freeze backbone (ignored if training full model at once)
freeze_epochs = 25

# Number of train epochs between evaluations on val data
val_interval = 5

# Number of epochs/evaluations without train/val improvement before stopping
stopping_intervals = (50, 10)


# Model configuration
model_name = "shufflenet_v2_x0_5"
model_weights = ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1
# If using locally-saved weights
model_path = None


# Optimizer configuration
optimizer_class = AdamW
optimizer_kwargs: dict[str, dict[str, Any]] = {
    "backbone": {
        "lr": 0.001,
        "betas": (0.9, 0.999),
        "weight_decay": 0.0001,
        "eps": 1e-8,
    },
    "head": {
        "lr": 0.1,
        "betas": (0.9, 0.999),
        "weight_decay": 0.0001,
        "eps": 1e-8,
    },
    "general": {"fused": True},
}


# Scheduler configuration
scheduler_class = lr_scheduler.OneCycleLR
scheduler_kwargs = {
    "max_lr": [
        optimizer_kwargs["backbone"]["lr"],
        optimizer_kwargs["head"]["lr"],
    ],
    "pct_start": 0.1,
    "anneal_strategy": "cos",
    "cycle_momentum": True,
    "base_momentum": 0.85,
    "max_momentum": 0.95,
    "div_factor": 25,
    "final_div_factor": 1e4,
    "three_phase": False,
    "last_epoch": -1,
}
scheduler1_class = None
scheduler1_kwargs: dict[str, Any] = {}
scheduler2_class = None
scheduler2_kwargs: dict[str, Any] = {}


# Loss function configuration

# Specify the amount of loss class balancing
loss_weights_balancing_factor = 0.0

loss_fn_class = MultiClassFocalLoss
# If you want to use the loss weights from above, set alpha to "weights"
loss_fn_kwargs: dict[str, int | float | str | Tensor] = {
    "gamma": 2,
    "alpha": "weights",
    "label_smoothing": 0.2,
    "focus_on": 1,
}


# Batch-level transforms setup
cutmix_args = {"p": 0.35, "alpha": 1.0}
mixup_args = {"p": 0.35, "alpha": 0.2}


# Initialization (do not make changes below when configuring training)
###############################################################################
###############################################################################

# General initialization
###############################################################################

# Set seed for reproducability
if seed == "default":
    seed = DEFAULT_SEED
manual_seed(seed)

# Replace device if device is "default"
if device == "default":
    device = get_device()

# Data initialization
###############################################################################

# Create PyTorch datasets/dataloaders
train_dataset, train_dataloader, val_dataset, val_dataloader = (
    make_training_dataloaders(
        train_data_path=train_path,
        train_dataloader_params=dataloader_params,
        train_transforms=train_transforms,
        val_data_path=val_path,
        val_dataloader_params=dataloader_params,
        val_transforms=val_transforms,
        loss_weights_balancing_factor=loss_weights_balancing_factor,
    )
)

input_dims = list(train_dataset[0][0].shape)

if val_path is not None:  # type: ignore
    logger.info_(
        "Created training and validation dataloaders using data at %s and %s",
        train_path,
        val_path,
    )
else:
    logger.info_("Created training dataloader using data at %s", train_path)


# Model initialization
###############################################################################

# Instantiate model and freeze backbone by default
if model_name == "shufflenet_v2_x0_5":  # type: ignore
    model = shufflenet_v2_x0_5(weights=model_weights)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = Linear(model.fc.in_features, len(train_dataset.classes))
    init.normal_(model.fc.weight)
    init.zeros_(model.fc.bias)
elif model_name == "mobilenet_v3_small":  # type: ignore
    model = mobilenet_v3_small(weights=model_weights)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier[3] = Linear(
        model.classifier[3].in_features,  # type: ignore
        len(train_dataset.classes),
    )
    init.normal_(model.classifier[3].weight)  # type: ignore
    init.zeros_(model.classifier[3].bias)  # type: ignore
elif model_name == "convnext_tiny":  # type: ignore
    model = convnext_tiny(weights=model_weights)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier[2] = Linear(
        model.classifier[2].in_features,  # type: ignore
        len(train_dataset.classes),
    )
    init.normal_(model.classifier[2].weight)  # type: ignore
    init.zeros_(model.classifier[2].bias)  # type: ignore
elif model_name == "efficientnet_v2_s":  # type: ignore
    model = efficientnet_v2_s(weights=model_weights)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier[1] = Linear(
        model.classifier[1].in_features,  # type: ignore
        len(train_dataset.classes),
    )
    init.normal_(model.classifier[1].weight)  # type: ignore
    init.zeros_(model.classifier[1].bias)  # type: ignore
elif model_name == "vit_b_32":  # type: ignore
    model = vit_b_32(weights=model_weights)
    for param in model.parameters():
        param.requires_grad = False
    model.heads.head = Linear(
        model.heads.head.in_features,  # type: ignore
        len(train_dataset.classes),
    )
    init.normal_(model.heads.head.weight)
    init.zeros_(model.heads.head.bias)
elif model_name == "maxvit_t":  # type: ignore
    model = maxvit_t(weights=model_weights)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier[5] = Linear(
        model.classifier[5].in_features,  # type: ignore
        len(train_dataset.classes),
    )
    init.normal_(model.classifier[5].weight)  # type: ignore
    init.zeros_(model.classifier[5].bias)  # type: ignore
elif model_name == "swin_t":  # type: ignore
    model = swin_t(weights=model_weights)
    for param in model.parameters():
        param.requires_grad = False
    model.head = Linear(model.head.in_features, len(train_dataset.classes))
    init.normal_(model.head.weight)
    init.zeros_(model.head.bias)
else:
    raise ValueError(f"Add {model_name} to model options before using")

# Set freeze_epochs to unfreeze backbone at start of training if needed
if len(optimizer_kwargs["head"]) == 0:  # type: ignore
    freeze_epochs = 0

# Load pre-trained weights if needed
if model_path is not None:
    model.load_state_dict(torch_load(model_path, weights_only=True))

# Place model on training device
model = model.to(device=device)

logger.info_("Created model on %s:\n%s", device.upper(), model)


# Optimization initialization
###############################################################################

# Set up optimizer

optimizer = optimizer_class(
    [
        {
            "params": [p for p in model.parameters() if not p.requires_grad],
            **optimizer_kwargs["backbone"],
        },
        {
            "params": [p for p in model.parameters() if p.requires_grad],
            **optimizer_kwargs["head"],
        },
    ],
    **optimizer_kwargs["general"],  # type: ignore
)


# Set up learning rate schedulers

schedulers: list[lr_scheduler.LRScheduler] = []

# pylint: disable=not-callable
if scheduler_class is lr_scheduler.OneCycleLR:
    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        epochs=total_epochs,
        steps_per_epoch=len(train_dataloader),
        **scheduler_kwargs,  # type: ignore
    )
    schedulers.append(scheduler)
elif scheduler_class is not None:  # type: ignore
    scheduler = scheduler_class(optimizer, **scheduler_kwargs)  # type: ignore
    schedulers.append(scheduler)
else:
    scheduler1 = scheduler1_class(  # type: ignore
        optimizer, **scheduler1_kwargs
    )
    scheduler2 = scheduler2_class(  # type: ignore
        optimizer, **scheduler2_kwargs
    )
    schedulers.append(scheduler1)
    schedulers.append(scheduler2)
# pylint: enable=not-callable


# Gather optimization components
optimization = [optimizer] + schedulers


# Create loss function

# Replace alpha=weights with the actual weights
if "alpha" in loss_fn_kwargs and loss_fn_kwargs["alpha"] == "weights":
    loss_fn_kwargs["alpha"] = train_dataset.loss_weights.to(device=device)
# Set up loss function
loss_fn = loss_fn_class(**loss_fn_kwargs)  # type: ignore


# Set up in-loop transforms
cutmix = CutMix(
    alpha=cutmix_args["alpha"], num_classes=len(train_dataset.classes)
)
mixup = MixUp(
    alpha=mixup_args["alpha"], num_classes=len(train_dataset.classes)
)
# Calculate probabilities for CutMix, MixUp, and no transform
inloop_probs = [
    cutmix_args["p"] / (cutmix_args["p"] + mixup_args["p"]),
    mixup_args["p"] / (cutmix_args["p"] + mixup_args["p"]),
    1 - cutmix_args["p"] - mixup_args["p"],
]
cutmix_or_mixup = RandomChoice([cutmix, mixup], p=inloop_probs[:-1])
inloop_transforms = RandomApply([cutmix_or_mixup], p=inloop_probs[-1])


# Set up trainer
trainer = Trainer(
    loaders=[train_dataloader, val_dataloader],  # type: ignore
    optimization=optimization,  # type: ignore
    loss_fn=loss_fn,
    inloop_transforms=inloop_transforms,
    automatic_mixed_precision=automatic_mixed_precision,
)

logger.info_("Created trainer with optimizers, schedulers, and loss function")

# automatic mixed precision not supported on mps, disabled in Trainer.train()
if automatic_mixed_precision and device == "mps":
    logger.info_(
        "Automatic mixed precision training is not supported on MPS and "
        "is being automatically disabled."
    )


# Run Training
###############################################################################
###############################################################################

logger.info_("Beginning training...")

model, n_steps_trained, n_epochs_trained = trainer.train(  # type: ignore
    model=model,
    n_epochs=total_epochs,
    val_interval=val_interval,
    freeze_epochs=freeze_epochs,
    stopping_intervals=stopping_intervals,
)

logger.info_("Finished running classifier training!")

# Archive logs and clean up checkpoints
# TODO: implement this functionality
