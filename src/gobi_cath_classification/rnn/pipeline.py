from collections import Counter
import h5py
from pathlib import Path
from gobi_cath_classification.pipeline.data.data_loading import read_in_embeddings
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from torch.nn.functional import one_hot

REPO_ROOT_DIR = Path(__file__).parent.parent.parent.parent.absolute()
DATA_DIR = REPO_ROOT_DIR / "data"

h5_file = h5py.File(DATA_DIR / "t5_xl_v3_half_cath_S100_RESIDUE.h5")
keys = list(h5_file.keys())
keys = [keys[i:i + 10] for i in range(0, len(keys), 10)]
print(keys)
print(type(keys[0]))
