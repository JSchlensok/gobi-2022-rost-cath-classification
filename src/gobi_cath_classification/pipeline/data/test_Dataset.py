from copy import deepcopy
import numpy as np
from sklearn.utils import shuffle

from ..data import Dataset
from ..utils import CATHLabel

x1 = np.array([[1], [2], [5]])
x2 = np.array([[1], [2], [6]])
id1, id2, id3, id4, id5, id6 = [
    "1.8.20.300",
    "2.4.6.8",
    "5.5.5.5",
    "6.6.6.6",
    "2.2.3.3",
    "5.5.5.5000",
]

data1 = Dataset(
    X_train=x1,
    y_train=[CATHLabel(label) for label in [id1, id2, id3]],
    train_labels=[CATHLabel(label) for label in [id1, id2, id3]],
    X_val=x2,
    y_val=[CATHLabel(label) for label in [id1, id2, id4]],
    X_test=x1,
    y_test=[CATHLabel(label) for label in [id1, id5, id6]],
    X_tmp_holdout=None,
    y_tmp_holdout=None,
)


data1_with_strings = deepcopy(data1)
data1_with_strings.load_strings(["1", "2", "5"], ["1", "2", "6"], ["1", "2", "5"], [])


def test_string_representation():
    assert data1_with_strings.get_split("train", "string")[0] == ["1", "2", "5"]
    assert data1_with_strings.get_split("val", "string")[0] == ["1", "2", "6"]


data1 = data1.get_filtered_version("H")


class TestFilteringForHLevel:
    def test_val(self, allclose):
        X, y = data1.get_split("val", "embedding-array")
        assert allclose(X, np.array([[1], [2]]))
        assert y == [id1, id2]

    def test_test(self, allclose):
        X, y = data1.get_split("test", "embedding-array")
        assert allclose(X, np.array([[1]]))
        assert y == [id1]


data2 = Dataset(
    X_train=np.array([[1], [2]]),
    y_train=[CATHLabel("1.8.20.300"), CATHLabel("2.20.2.2")],
    train_labels=[CATHLabel("1.8.20.300"), CATHLabel("2.20.2.2")],
    X_val=np.array([[3], [4]]),
    y_val=[CATHLabel("1.8.20.300"), CATHLabel("2.2.2.2")],
    X_test=np.array([[5], [6]]),
    y_test=[CATHLabel("1.8.20.300"), CATHLabel("2.20.3.3")],
    X_tmp_holdout=None,
    y_tmp_holdout=None,
)

data2 = data2.get_filtered_version("A")


class TestFilteringForALevel:
    def test_val(self, allclose):
        X, y = data2.get_split("val")
        assert allclose(X, np.array([[3]]))
        assert y == ["1.8.20.300"]

    def test_test(self, allclose):
        X, y = data2.get_split("test")
        assert allclose(X, np.array([[5], [6]]))
        assert y == ["1.8.20.300", "2.20.3.3"]


def test_shuffling(allclose):
    shuffled_data = data1.get_copy()
    shuffled_data.shuffle_training_set(np.random.RandomState(42))
    assert allclose(
        shuffled_data.X_train, shuffle(data1.X_train, random_state=np.random.RandomState(42))
    )
