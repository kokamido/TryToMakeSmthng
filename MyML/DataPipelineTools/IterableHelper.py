from typing import Iterable, TypeVar, List

T = TypeVar('T')


def get_n_or_less_elements(data: Iterable[T], n: int) -> List[T]:
    res = []
    for index, element in enumerate(data):
        if index == n:
            break
        res.append(element)
    return res
