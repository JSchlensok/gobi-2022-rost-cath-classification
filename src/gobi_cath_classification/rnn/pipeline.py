from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from torch.nn.functional import one_hot

REPO_ROOT_DIR = Path(__file__).parent.parent.parent.parent.absolute()
DATA_DIR = REPO_ROOT_DIR / "data"

