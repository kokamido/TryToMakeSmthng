from abc import ABC, abstractmethod
from typing import Tuple, Any, Union, List

import numpy as np
from numpy.typing import ArrayLike


class DataLoader(ABC):

    @abstractmethod
    def get_data_batch(self, batch_size: int) -> Tuple[ArrayLike, ArrayLike]:
        pass


class NumpyDataLoader(DataLoader):

    def __init__(self, data: ArrayLike, labels: ArrayLike, shuffle: bool = False):
        permutation = np.random.permutation(data.shape[0]) if shuffle else np.arange(data.shape[0])
        self.__data__ = data.copy()[permutation]
        self.__labels__ = np.array(labels.copy())[permutation]
        self.__current_index__ = 0

    def get_data_batch(self, batch_size: int) -> Tuple[ArrayLike, ArrayLike]:
        left = self.__current_index__
        self.__current_index__ += batch_size
        return self.__data__[left: self.__current_index__,], self.__labels__[left: self.__current_index__]

