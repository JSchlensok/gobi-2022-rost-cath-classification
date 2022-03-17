from pathlib import Path

import numpy as np
from pytorch_metric_learning import losses, trainers
import sklearn
import torch
from tqdm import tqdm

from gobi_cath_classification.pipeline.data import data_loading
from gobi_cath_classification.pipeline.utils.torch_utils import get_device
from gobi_cath_classification.contrastive_learning_arcface.FNN import FNN

MODEL_DIR = Path(
    "G:\\My Drive\\Files\\Projects\\University\\2021W\\GoBi\\Project\\gobi-2022-rost-cath-classification\\models"
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

    # TODO move to Dataset class
    train_X, train_y = dataset.get_split("train", x_encoding="embedding-tensor", zipped=False)
    label_encoder = sklearn.preprocessing.LabelEncoder()
    train_y_encoded = torch.as_tensor(
        label_encoder.fit_transform([str(label) for label in train_y])
    )
    train_data = list(zip(train_X, train_y_encoded))

    # TODO set batch size

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

    for epoch in range(1, num_epochs + 1):
        tdata = tqdm(train_data)
        for i, (X, y) in enumerate(tdata):
            tdata.set_description(f"Epoch {epoch}")
            X = X.to(device)
            X_embedded = torch.unsqueeze(model(X), 0)
            y = torch.unsqueeze(y, 0)

            fnn_optimizer.zero_grad()
            fnn_loss = criterion(X_embedded, y)
            fnn_loss.backward()
            fnn_optimizer.step()
            loss_optimizer.step()

        # TODO dynamically set save path
        if epoch % 5 == 0:
            torch.save(model.fnn.state_dict(), MODEL_DIR / f"arcface_2022-02-24_epoch_{epoch}.pth")



if __name__ == "__main__":
    train(100)
