from .CATHLabel import CATHLabel


def test_indexing():
    label = CATHLabel("1.400.45.200")
    assert label["C"] == "1"
    assert label["A"] == "400"

def test_string_return():
    label = CATHLabel("1.400.45.200")
    assert isinstance(label["C"], str)
    assert isinstance(label["A"], str)

def test_slicing():
    label = CATHLabel("1.400.45.200")
    assert label[:"A"] == "1.400"
    assert label[:"T"] == "1.400.45"
    assert label[:"H"] == "1.400.45.200"
