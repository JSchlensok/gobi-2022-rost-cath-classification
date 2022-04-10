from collections import Counter
from pathlib import Path
from gobi_cath_classification.pipeline.data.data_loading import read_in_embeddings
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from torch.nn.functional import one_hot

REPO_ROOT_DIR = Path(__file__).parent.parent.parent.parent.absolute()
DATA_DIR = REPO_ROOT_DIR / "data"

emb = read_in_embeddings(DATA_DIR / "t5_xl_v3_half_cath_S100_RESIDUE.h5")
print(f"Länge: {len(emb)}")
print(emb.keys()[0])
print(emb.values()[0].shape)
