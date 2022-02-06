from collections import Counter
from typing import List

import numpy as np


def compute_inverse_sample_weights(labels: List[str]) -> np.ndarray:
    counts = Counter(labels)
    sample_weights = np.array([1 / counts[label] for label in labels])
    return sample_weights
