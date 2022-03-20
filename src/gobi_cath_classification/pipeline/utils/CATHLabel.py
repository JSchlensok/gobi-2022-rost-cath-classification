from __future__ import annotations
from functools import total_ordering
from typing import Union
from typing_extensions import Literal


@total_ordering
class CATHLabel:
    """Utility class to take advantage of the hierarchical CATH indices

    Uses:
        label[:level] (level ∈ "CATH") returns the ID up to that level (inclusive)

        label[level] (level ∈ "CATH") returns the ID of the specific level as a string
            (since it encodes no more hierarchical information)

        Labels can be sorted lexicographically

        Labels can be compared to other labels or directly to strings


    """

    def __init__(self, label: str):
        self._string = label
        self._levels = label.split(".")

    def __len__(self) -> int:
        return len(self._levels)

    def __repr__(self):
        return self._string

    def __str__(self):
        return self._string

    def __eq__(self, other):
        if isinstance(other, str):
            return self._string == other

        elif not isinstance(other, CATHLabel):
            return False

        else:
            return self._string == other._string

    def __lt__(self, other):
        if not isinstance(other, CATHLabel):
            return False

        return self._levels < other._levels

    def __hash__(self):
        return self._string.__hash__()

    def __getitem__(self, key: Union[Literal["C", "A", "T", "H"], slice]) -> Union[CATHLabel, str]:
        if isinstance(key, slice):
            if isinstance(key.start, int) or isinstance(key.stop, int):
                raise KeyError("Int indices not supported for CATH labels")
            elif key.start is None:
                return CATHLabel(".".join(self._levels[: "CATH".index(key.stop) + 1]))
            elif isinstance(key.start, str) and key.start in "CATH":
                return CATHLabel(
                    ".".join(self._levels["CATH".index(key.step) : "CATH".index(key.stop) + 1])
                )
            elif key.step is not None:
                raise KeyError("Step size not supported for CATH labels")

        return self._levels["CATH".index(key)]
