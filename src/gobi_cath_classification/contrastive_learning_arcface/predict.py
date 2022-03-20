import logging
from pathlib import Path
import pickle
from typing import List, Tuple

import numpy as np
import torch
from torch.nn.functional import cosine_similarity

from gobi_cath_classification.pipeline.data.data_loading import load_data
from gobi_cath_classification.pipeline.evaluation import evaluate, accuracy_for_level
from gobi_cath_classification.pipeline.utils import CATHLabel
from gobi_cath_classification.contrastive_learning_arcface.FNN import FNN
from gobi_cath_classification.contrastive_learning_arcface.utils import get_base_dir
from gobi_cath_classification.pipeline.utils.torch_utils import get_device

ROOT_DIR = get_base_dir()
MODEL_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"

logging.basicConfig(level=logging.DEBUG)


def transfer_annotations(
    lookup: torch.Tensor, query: torch.Tensor, lookup_labels: List[CATHLabel]
) -> CATHLabel:
    """Performs EAT given a set of labelled lookup protein embeddings and a query protein embedding"""

    distances = lookup.float() - query.float().unsqueeze(dim=0)
    closest_neighbor_index = torch.linalg.norm(
        distances, dim=1
    ).argmin()  #  TODO change to cosine similarity

    # TODO prevent self-lookup

    return lookup_labels[closest_neighbor_index]


def main() -> None:
    # TODO parse args
    logging.debug("Loading dataset")
    dataset = load_data(DATA_DIR, np.random.RandomState(42), True, True, False, True)

    """
    # Load embeddings
    id2embedding = read_in_embeddings(DATA_DIR / "cath_v430_dom_seqs_S100_161121.h5")
    id2label = read_in_labels(DATA_DIR / "cath-domain-list.txt")
    
    # Load lookup
    id2seq_lookup = read_in_sequences(DATA_DIR / "train74k.fasta")
    lookup_labels = [id2label[id] for id in id2seq_lookup.keys()]

    # Load query sequences
    id2seq_query = 
    query = None
    """
    logging.debug("Getting splits from dataset")
    lookup_embeddings, lookup_labels = dataset.get_split(
        "train", x_encoding="embedding-tensor", zipped=False
    )

    query_embeddings, query_labels = dataset.get_split(
        "test", x_encoding="embedding-tensor", zipped=False
    )

    logging.debug("Loading model")
    model_name = f"arcface_2022-03-20_epoch_50.pth"
    device = get_device()
    model = FNN.from_file(MODEL_DIR / model_name).to(device)
    model = FNN().to(device)

    """
    lookup_embedding_path = Path(DATA_DIR / "train_set_lookup_embeddings_50epochs")
    if not lookup_embedding_path.exists():
        logging.debug("Generating lookup embeddings")
        lookup_embeddings = model(lookup_embeddings)
        with open(lookup_embedding_path, "wb+") as f:
            pickle.dump(lookup_embeddings, f)
    else:
        logging.debug("Loading serialized lookup embeddings")
        with open(lookup_embedding_path, "wb+") as f:
            lookup_embeddings = pickle.load(f)
    """

    lookup_embeddings = lookup_embeddings.cuda()
    query_embeddings = query_embeddings.cuda()

    with torch.cuda.amp.autocast():
        lookup_embeddings = model(lookup_embeddings)
        query_embeddings = model(query_embeddings)

    logging.debug("Transferring annotations")
    predictions = [
        transfer_annotations(lookup_embeddings, query_embedding, lookup_labels)
        for query_embedding in query_embeddings
    ]

    # Analyze
    logging.debug("Evaluating predictions")
    str_preds = [str(pred) for pred in predictions]
    for level in "CATH":
        print(level)
        print(accuracy_for_level(query_labels, str_preds, dataset.train_labels, level))


if __name__ == "__main__":
    main()
