""" Script to run classifier training. """

# pylint: disable=unused-import
from torch.nn import CrossEntropyLoss, Linear, Sequential
from torch.optim import AdamW, lr_scheduler
from torchvision.models import (
    EfficientNet_B3_Weights,
    ShuffleNet_V2_X0_5_Weights,
    resnet18,
    shufflenet_v2_x0_5,
    squeezenet1_1,
)
from torchvision.transforms.v2 import (
    CenterCrop,
    GaussianBlur,
    GaussianNoise,
    RandomHorizontalFlip,
    RandomPerspective,
    RandomResizedCrop,
    Resize,
)

from maviratrain.data.classification_dataset import make_training_dataloaders
from maviratrain.models.classifier_model import (
    TestModel,
    create_efficientnet_b3,
    create_vit,
)
from maviratrain.train.train_classifier import Trainer
from maviratrain.utils.general import get_device, get_logger


# set up logger
logger = get_logger(
    "scripts.run_classifier_training",
    log_filename=("../logs/train_runs/classifier/run_classifier_training.log"),
    rotation_params=(1000000, 1000),  # 1 MB, 1000 backups
)

logger.info_("Beginning setup for training...")

# Create datasets and dataloaders

# The path to the dataset used for training
train_path = (
    "/home/danny/mavira/FashionTraining/data/classifier/"
    "classifier362-r1-s1-n1/train/"
)
# train_path = "/mnt/disks/localssd/data/classifier362-r2-s3-n4/train/"

# The path to the dataset used for validation
val_path = (
    "/home/danny/mavira/FashionTraining/data/classifier/"
    "classifier362-r1-s1-n1/val/"
)
# val_path = "/mnt/disks/localssd/data/classifier362-r2-s3-n4/val/"

# Specify any non-default dataloader parameters
# Defaults found in maviratrain.utils.constants
dataloader_params = {"batch_size": 128, "num_workers": 12}

# Specify any additional transforms to apply to the data
transforms = Sequential(
    RandomHorizontalFlip(p=0.5),
    RandomResizedCrop(size=224, scale=(0.25, 1.0)),
    GaussianNoise(mean=0.0, sigma=0.1, clip=False),
    RandomPerspective(distortion_scale=0.5, p=0.5),
)

# Specify the amount of loss balancing
loss_weights_balancing_factor = 1.0

# Create PyTorch datasets/dataloaders
train_dataset, train_dataloader, val_dataset, val_dataloader = (
    make_training_dataloaders(
        train_data_path=train_path,
        transforms=transforms,
        train_dataloader_params=dataloader_params,
        val_data_path=val_path,
        val_dataloader_params=dataloader_params,
        loss_weights_balancing_factor=loss_weights_balancing_factor,
    )
)

input_dims = list(train_dataset[0][0].shape)

# TODO: currently ignoring type because we are just manually modifying the file
if val_path is not None:  # type: ignore
    logger.info_(
        "Created training and validation dataloaders using data at %s and %s",
        train_path,
        val_path,
    )
else:
    logger.info_("Created training dataloader using data at %s", train_path)


# Create a model

# Create ViT model
# vit_kwargs = {
#     "image_size": train_dataset[0][0].shape[-1],
#     "patch_size": 32,
#     "num_layers": 8,
#     "num_heads": 8,
#     "hidden_dim": 320,
#     "mlp_dim": 1024,
#     "dropout": 0.2,
#     "attention_dropout": 0.2,
#     "representation_size": None,
#     "num_classes": len(train_dataset.classes),
# }
# model = create_vit(vit_kwargs=vit_kwargs)

# Create EfficientNet-B3 model
# model = create_efficientnet_b3(
#     # weights=None,
#     weights=EfficientNet_B3_Weights.IMAGENET1K_V1,
#     num_classes=len(train_dataset.classes),
# )

# Resume training an EfficientNet-B3 model from a checkpoint
# model = create_efficientnet_b3(
# weights=(
# "/home/danny/mavira/FashionTraining/checkpoints/"
# "temp24_S42816_A8.131024986505508_T2025-01-15T02:06:36+00:00.pt"
# ),
# num_classes=len(train_dataset.classes),
# )

# Create ShuffleNetV2 model
# model = shufflenet_v2_x0_5(num_classes=len(train_dataset.classes)).to("cuda")
model = shufflenet_v2_x0_5(weights=ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1)
model.fc = Linear(
    in_features=model.fc.in_features, out_features=len(train_dataset.classes)
)
model = model.to("cuda")

# Create SqueezeNet 1.1 model
# model = squeezenet1_1(num_classes=len(train_dataset.classes)).to("cuda")

# Create ResNet-18 model
# model = resnet18(num_classes=len(train_dataset.classes)).to("cuda")

# Create TestModel model
# model = TestModel(
#     input_dims=input_dims, num_classes=len(train_dataset.classes)
# )

device = get_device().upper()

logger.info_("Created model on %s:\n%s", device, model)


# Create optimizers, learning rate schedulers, and loss function for trainer

# Number of epochs for warmup and total training
warmup_epochs = 10
total_epochs = 500

# Set up optimizer
optimizer = AdamW(
    params=model.parameters(),
    lr=0.001,  # 0.001 originally
    betas=(0.9, 0.999),
    eps=1e-10,
    weight_decay=0.01,  # 0.01 originally
    fused=True,
)

# Set up learning rate schedulers
scheduler1 = lr_scheduler.LinearLR(
    optimizer,
    start_factor=0.1,
    end_factor=1,
    total_iters=warmup_epochs,
    last_epoch=-1,
)
scheduler2 = lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=total_epochs, last_epoch=warmup_epochs
)

# Gather optimization components
optimization = [optimizer, scheduler1, scheduler2]


# Create loss function
loss_fn = CrossEntropyLoss(
    weight=None,  # train_dataset.standard_loss_weights.to("cuda"),  # TODO
    reduction="sum",
    label_smoothing=0.8,
)

# Set up trainer
trainer = Trainer(
    loaders=[train_dataloader, val_dataloader],  # type: ignore
    optimization=optimization,  # type: ignore
    loss_fn=loss_fn,
)

logger.info_("Created trainer with optimizers, schedulers, and loss function")


# Train model

logger.info_("Beginning training...")

model, n_steps_trained, n_epochs_trained = trainer.train(  # type: ignore
    model=model, n_epochs=total_epochs, val_interval=5
)

logger.info_("Training complete!")

# Archive logs and clean up checkpoints
# TODO: implement this functionality
