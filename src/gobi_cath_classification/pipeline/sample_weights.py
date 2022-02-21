from collections import Counter, OrderedDict
from typing import List

import numpy as np


def compute_class_weights(labels: List[str]) -> np.ndarray:
    """
    Computes class weights.

    Example:
        Input:  ["dog", "cat", "dog", "dog", "dog]
        Output: np.array([1, 0.25])
                (class weight for cat = 1,
                 class weight for dog = 0.25)
    """
    counts_labels = Counter(labels)
    class_weights = np.array([1 / value for key, value in sorted(counts_labels.items())])
    assert len(class_weights) == len(list(set(labels)))
    return class_weights


def compute_inverse_sample_weights(labels: List[str]) -> np.ndarray:
    """
    Computes class weights and returns an array of class weight for each label in labels.

     Example:
        Input:  ["dog", "cat", "dog", "dog", "dog"]
        Output: np.array([0.25, 1, 0.25, 0.25, 0.25])

    """
    counts = Counter(labels)
    sample_weights = np.array([1 / counts[label] for label in labels])
    assert len(labels) == len(sample_weights)
    return sample_weights


def compute_class_counts(labels: List[str]) -> np.ndarray:
    """
    Computes the counts for each label and returns them in alphabetical order according to their label names

    Example:
        Input: ["dog", "cat", "dog", "dog", "dog"]
        Output: np.array([1, 4])

    """
    label_count = {}
    for label in labels:
        if label in label_count:
            label_count[label] += 1
        else:
            label_count[label] = 1

    ordered_label = OrderedDict(sorted(label_count.items()))
    return np.array([count for label, count in ordered_label.items()])
