from .CATHLabel import CATHLabel


def test_label_slicing():
    label = CATHLabel("1.400.45.200")
    assert label["C"] == "1"
    assert label["A"] == "1.400"
    assert label["T"] == "1.400.45"
    assert label["H"] == "1.400.45.200"