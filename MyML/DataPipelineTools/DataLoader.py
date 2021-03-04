from abc import ABC, abstractmethod
from typing import Any, Callable, Tuple

import numpy as np


class DataLoader(ABC):
    @abstractmethod
    def get_data_batch(self, batch_size: int) -> Any:
        """
        :param batch_size: how many data do you want
        :return: batch of data
        """
        pass


class NumpyDataLoader(DataLoader):
    def __init__(self, data: np.ndarray, labels: np.ndarray, shuffle: bool = False):
        permutation = (
            np.random.permutation(data.shape[0])
            if shuffle
            else np.arange(data.shape[0])
        )
        self.__data__ = data.copy()[permutation]
        self.__labels__ = np.array(labels.copy())[permutation]
        self.__current_index__ = 0

    def get_data_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        left = self.__current_index__
        self.__current_index__ += batch_size
        return (
            self.__data__[
                left : self.__current_index__,
            ],
            self.__labels__[left : self.__current_index__],
        )


class GeneratorBasedLoader(DataLoader):
    def __init__(self, generator_func: Callable[[], Tuple[Any, Any]]):
        """
        :param generator_func:
        :param calc_shapes_on_init:
        """
        self.__generator_func__ = generator_func

    def get_data_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        data = []
        target = []
        for _ in range(batch_size):
            current_data, current_target = self.__generator_func__()
            data.append(current_data)
            target.append(current_target)
        return np.array(data), np.array(target)


class StubDataLoader(DataLoader):
    def __init__(self):
        self.__stub__ = np.array([0])

    def get_data_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.__stub__, self.__stub__
