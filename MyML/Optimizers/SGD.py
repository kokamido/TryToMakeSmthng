import numpy as np

from .Optimizer import Optimizer
from ..CalcGraph.AbstractGraph import CalcGraph
from ..DataPipelineTools.DataLoader import DataLoader
from ..Losses.AbstractLoss import AbstractLoss


class SGDOptimizer(Optimizer):
    def __init__(
        self, learning_rate: float, max_batch_size: int, norm_grad: bool = False
    ):
        self.__learning_rate__ = learning_rate
        self.__max_batch_size__ = max_batch_size
        self.__norm_grad__ = norm_grad

    def update_node_parameters(
        self, graph_node: CalcGraph, loss: AbstractLoss, data: DataLoader
    ) -> None:
        parameters_to_update = graph_node.get_learnable_parameters()
        data_to_calc_grads, labels = data.get_data_batch(self.__max_batch_size__)
        grad = np.squeeze(
            np.apply_along_axis(graph_node.calc_grads, -1, data_to_calc_grads)
        )
        if self.__norm_grad__:
            grad /= np.linalg.norm(grad)
        predicted_values = np.squeeze(
            np.apply_along_axis(graph_node.calc_forward, -1, data_to_calc_grads)
        )
        loss_grads = np.apply_along_axis(
            loss.calc_grads, -1, np.dstack((predicted_values, labels))
        )
        loss_grads = np.tile(np.squeeze(loss_grads), (parameters_to_update.size, 1)).T
        grad *= loss_grads
        grad = grad.mean(axis=0)
        parameters_to_update -= self.__learning_rate__ * grad
