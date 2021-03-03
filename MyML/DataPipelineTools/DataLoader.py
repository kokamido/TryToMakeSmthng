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

    @abstractmethod
    def get_data_and_target_shapes(self) -> Tuple[Tuple[int], Tuple[int]]:
        """
        :return: (shape of single data item, shape of single target item)
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
        self.__data_shape__ = self.__data__[0].shape
        self.__target_shape__ = self.__labels__[0].shape

    def get_data_and_target_shapes(self) -> Tuple[Tuple[int], Tuple[int]]:
        return self.__data_shape__, self.__target_shape__

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
    """
    Should bu used for test cases only
    """

    def __init__(
        self, generator_func: Callable[[], Tuple[Any, Any]], calc_shapes_on_init=False
    ):
        self.__generator_func__ = generator_func
        self.__data_shape__ = None
        self.__target_shape__ = None
        if calc_shapes_on_init:
            self.get_data_batch(1)

    def get_data_and_target_shapes(self) -> Tuple[Tuple[int], Tuple[int]]:
        if self.__data_shape__ is None or self.__target_shape__ is None:
            raise Exception(
                "Must call 'get_data_batch' method almost once before 'get_data_and_target_shapes' call"
            )
        return self.__data_shape__, self.__target_shape__

    def get_data_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        data = []
        target = []
        for _ in range(batch_size):
            current_data, current_target = self.__generator_func__()
            data.append(current_data)
            target.append(current_target)
        data_arr, target_arr = np.array(data), np.array(target)
        if self.__target_shape__ is None or self.__data_shape__ is None:
            self.__target_shape__ = (1,) if np.isscalar(target[0]) else target[0].shape
            self.__data_shape__ = (1,) if np.isscalar(data[0]) else data[0].shape
        return data_arr, target_arr
