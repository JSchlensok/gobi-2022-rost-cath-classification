from copy import deepcopy
import numpy as np

from .Dataset import Dataset
from ..utils.CATHLabel import CATHLabel

x1 = np.array([[1], [2], [5]])
x2 = np.array([[1], [2], [6]])
id1, id2, id3, id4, id5, id6 = ["1.8.20.300", "2.4.6.8", "5.5.5.5", "6.6.6.6", "2.2.3.3", "5.5.5.5000"]

data1 = Dataset(
    X_train=x1,
    y_train=[CATHLabel(label) for label in [id1, id2, id3]],
    train_labels=[CATHLabel(label) for label in [id1, id2, id3]],
    X_val=x2,
    y_val=[CATHLabel(label) for label in [id1, id2, id4]],
    X_test=x1,
    y_test=[CATHLabel(label) for label in [id1, id5, id6]]
)

data1.filter("H")


class TestFilteringForHLevel:
    def test_val(self, allclose):
        X, y = data1.getSplit("val")
        assert allclose(X, np.array([[1], [2]]))
        assert y == [id1, id2]

    def test_test(self, allclose):
        X, y = data1.getSplit("test")
        assert allclose(X, np.array([[1]]))
        assert y == [id1]


data2 = Dataset(
    X_train=np.array([[1], [2]]),
    y_train=[CATHLabel("1.8.20.300"), CATHLabel("2.20.2.2")],
    train_labels=[CATHLabel("1.8.20.300"), CATHLabel("2.20.2.2")],
    X_val=np.array([[3], [4]]),
    y_val=[CATHLabel("1.8.20.300"), CATHLabel("2.2.2.2")],
    X_test=np.array([[5], [6]]),
    y_test=[CATHLabel("1.8.20.300"), CATHLabel("2.20.3.3")]
)

data2.filter("A")


class TestFilteringForALevel:
    def test_val(self, allclose):
        X, y = data2.getSplit("val")
        assert allclose(X, np.array([[3]]))
        assert y == ["1.8.20.300"]

    def test_test(self, allclose):
        X, y = data2.getSplit("test")
        assert allclose(X, np.array([[5], [6]]))
        assert y == ["1.8.20.300", "2.20.3.3"]
