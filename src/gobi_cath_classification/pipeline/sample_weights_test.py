import numpy as np

from gobi_cath_classification.pipeline.sample_weights import (
    compute_inverse_sample_weights,
    compute_class_weights
)


def test_compute_inverse_sample_weights():
    sample_weights = compute_inverse_sample_weights(labels=["cat", "dog", "giraffe", "cat"])
    np.testing.assert_allclose(sample_weights, np.array([0.5, 1, 1, 0.5]))
    # assert sample_weights.all() == np.array([0.5, 1, 1, 0.5]).all()


def test_compute_class_weights():
    y = ["dog", "dog", "cat", "dog", "dog"]
    class_weights = compute_class_weights(labels=y)
    np.testing.assert_allclose(class_weights, np.array([1, 0.25]))

