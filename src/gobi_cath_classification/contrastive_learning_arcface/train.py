from pathlib import Path

import numpy as np
from pytorch_metric_learning import losses, trainers
import sklearn
import torch
from tqdm import tqdm
from typing_extensions import Literal

# TODO use tqdm.rich

from gobi_cath_classification.pipeline.data import data_loading, Dataset
from gobi_cath_classification.pipeline.utils.torch_utils import get_device
from gobi_cath_classification.contrastive_learning_arcface.FNN import FNN
from gobi_cath_classification.contrastive_learning_arcface.utils import get_base_dir


def create_dataloader(
    dataset: Dataset, split: Literal["train", "val", "test"], batch_size: int
) -> torch.utils.data.DataLoader:
    train_X, train_y = dataset.get_split("train", x_encoding="embedding-tensor", zipped=False)
    label_encoder = sklearn.preprocessing.LabelEncoder()
    train_y_encoded = torch.as_tensor(
        label_encoder.fit_transform([str(label) for label in train_y])
    )

    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_X, train_y_encoded),
        batch_size=batch_size,
        shuffle=True,
    )


def train(num_epochs: int):
    dataset = data_loading.load_data(
        data_loading.DATA_DIR,
        rng=np.random.RandomState(42),
        without_duplicates=True,
        shuffle_data=False,
        load_only_small_sample=False,
        reloading_allowed=True,
    )

    BATCH_SIZE = 64
    train_dataloader = create_dataloader(dataset, "train", BATCH_SIZE)
    val_dataloader = create_dataloader(dataset, "val", BATCH_SIZE)

    # TODO tune margin & scale parameters
    criterion = losses.ArcFaceLoss(
        num_classes=len(dataset.train_labels), embedding_size=128, margin=28.6, scale=64
    )

    # TODO allow model reloading from savepoint
    device = get_device()
    model = FNN().to(device)

    # TODO tune fnn_optimizer parameters
    fnn_optimizer = torch.optim.Adam(model.parameters(), lr=10e-4, weight_decay=10e-4)
    loss_optimizer = torch.optim.Adam(criterion.parameters(), lr=10e-4, weight_decay=10e-4)

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(1, num_epochs + 1):
        running_loss = 0

        with tqdm(train_dataloader, unit="batch") as tepoch:
            for batch_id, (X_batch, y_batch) in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch}")

                X_batch = X_batch.cuda()
                y_batch = y_batch.cuda()
                fnn_optimizer.zero_grad()

                with torch.cuda.amp.autocast():
                    X_embedded = model(X_batch)
                    fnn_loss = criterion(X_embedded, y_batch)

                # TODO compute accuracy during training

                # TODO compute validation loss

                scaler.scale(fnn_loss).backward()
                running_loss += fnn_loss.item()

                train_loss = running_loss / (batch_id + 1)

                scaler.step(fnn_optimizer)
                scaler.update()

                tepoch.set_postfix(train_loss=f"{train_loss:.3f}")

            if epoch % 25 == 0:
                model_name = f"arcface_2022-03-20_epoch_{epoch}.pth"
                print("Checkpointing model to {model_name} ...")
                torch.save(
                    model.fnn.state_dict(),
                    get_base_dir() / f"models/{model_name}"
                )


if __name__ == "__main__":
    train(100)
