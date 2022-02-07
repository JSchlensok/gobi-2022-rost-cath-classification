import numpy as np
from sklearn.linear_model import LogisticRegression

from gobi_cath_classification.pipeline.data_loading import (
    DataSplits,
    label_for_level,
    load_data,
    DATA_DIR,
)


def test_label_for_level():
    label = "1.400.45.200"
    assert label_for_level(label=label, cath_level="C") == "1"
    assert label_for_level(label=label, cath_level="A") == "1.400"
    assert label_for_level(label=label, cath_level="T") == "1.400.45"
    assert label_for_level(label=label, cath_level="H") == "1.400.45.200"


def test_data_loading():
    # +
    # data_dir = Path(__file__).parent.parent.parent.parent / "data"
    # data_dir = Path("../data")

    dataset = load_data(data_dir=DATA_DIR, without_duplicates=False, shuffle_data=True)
    dataset_no_dup = load_data(
        data_dir=DATA_DIR, without_duplicates=True, shuffle_data=True
    )

    print(f"dataset.get_shape() = {dataset.get_shape()}")
    print(f"dataset_no_dup.get_shape() = {dataset_no_dup.get_shape()}")

    # print("Create model ...")
    # model = LogisticRegression()
    #
    # print("Train model ...")
    # model.fit(X=dataset_shuffled.X_train[:10000, :], y=dataset_shuffled.y_train[:10000])
    # model.fit(X=dataset.X_train, y=dataset.y_train)
    # model.score(X=dataset.X_test, y=dataset.y_test)


def test_get_set_for_level_H():
    data = DataSplits(
        X_train=np.array([[1], [2], [5]]),
        y_train=np.array(["1.8.20.300", "2.4.6.8", "5.5.5.5"]),
        X_val=np.array([[1], [2], [6]]),
        y_val=np.array(["1.8.20.300", "2.4.6.8", "6.6.6.6"]),
        X_test=np.array([[1], [2], [5]]),
        y_test=np.array(["1.8.20.300", "2.2.3.3", "5.5.5.5000"]),
        all_labels_train_sorted=["1.8.20.300", "2.4.6.8", "5.5.5.5"],
    )

    # val set
    np.testing.assert_allclose(
        actual=data._get_filtered_set_for_level(
            X=data.X_val, y=data.y_val, cath_level="H"
        )[0],
        desired=np.array([[1], [2]]),
    )
    y_val_at_level_3 = data._get_filtered_set_for_level(
        X=data.X_val, y=data.y_val, cath_level="H"
    )[1]
    np.testing.assert_equal(y_val_at_level_3, np.array(["1.8.20.300", "2.4.6.8"]))

    # test set
    np.testing.assert_allclose(
        actual=data._get_filtered_set_for_level(
            X=data.X_test, y=data.y_test, cath_level="H"
        )[0],
        desired=np.array([[1]]),
    )
    y_val_at_level_3 = data._get_filtered_set_for_level(
        X=data.X_test, y=data.y_test, cath_level="H"
    )[1]
    np.testing.assert_equal(y_val_at_level_3, np.array(["1.8.20.300"]))


def test_get_set_for_level_A():
    data = DataSplits(
        X_train=np.array([[1], [2]]),
        y_train=np.array(["1.8.20.300", "2.20.2.2"]),
        X_val=np.array([[3], [4]]),
        y_val=np.array(["1.8.20.300", "2.2.2.2"]),
        X_test=np.array([[5], [6]]),
        y_test=np.array(["1.8.20.300", "2.20.3.3"]),
        all_labels_train_sorted=["1.8.20.300", "2.20.2.2"],
    )

    # val set
    np.testing.assert_allclose(
        actual=data._get_filtered_set_for_level(
            X=data.X_val, y=data.y_val, cath_level="A"
        )[0],
        desired=np.array([[3]]),
    )
    y_val_at_level_3 = data._get_filtered_set_for_level(
        X=data.X_val, y=data.y_val, cath_level="A"
    )[1]
    np.testing.assert_equal(y_val_at_level_3, np.array(["1.8.20.300"]))

    # test set
    np.testing.assert_allclose(
        actual=data._get_filtered_set_for_level(
            X=data.X_test, y=data.y_test, cath_level="A"
        )[0],
        desired=np.array([[5], [6]]),
    )
    y_val_at_level_1 = data._get_filtered_set_for_level(
        X=data.X_test, y=data.y_test, cath_level="A"
    )[1]
    np.testing.assert_equal(y_val_at_level_1, np.array(["1.8.20.300", "2.20.3.3"]))
