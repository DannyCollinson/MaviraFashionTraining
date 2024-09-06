"""Testing PyTorch training"""

from torch.nn import CrossEntropyLoss
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader

import maviratrain.data.classification_dataset
import maviratrain.models.classifier_model
import maviratrain.train.train_classifier

if __name__ == "__main__":
    train_dataset = maviratrain.data.classification_dataset.ClassifierDataset(
        "data/classification73-resized_224x224-2024-08-30-split-2024-08-30/"
        "train"
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, drop_last=True
    )

    val_dataset = maviratrain.data.classification_dataset.ClassifierDataset(
        "data/classification73-resized_224x224-2024-08-30-split-2024-08-30/"
        "val"
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=128, shuffle=False, drop_last=False
    )

    vit_kwargs = {
        "image_size": 224,
        "patch_size": 32,
        "num_layers": 4,
        "num_heads": 3,
        "hidden_dim": 129,
        "mlp_dim": 256,
        "num_classes": 73,
    }
    model = maviratrain.models.classifier_model.create_vit(
        vit_kwargs=vit_kwargs
    )

    optimizer = AdamW(
        params=model.parameters(), lr=0.005, weight_decay=0.2, foreach=True
    )
    scheduler1 = lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1,
        total_iters=30,
        last_epoch=-1,
    )
    scheduler2 = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=330, last_epoch=30
    )
    optimization = [optimizer, scheduler1, scheduler2]

    trainer = maviratrain.train.train_classifier.Trainer(
        loaders=[train_dataloader, val_dataloader],
        optimization=optimization,
        loss_fn=CrossEntropyLoss(reduction="sum"),
    )

    model, n_steps_trained, n_epochs_trained = trainer.train(
        model=model, n_epochs=330
    )
