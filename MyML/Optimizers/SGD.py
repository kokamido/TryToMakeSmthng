import numpy as np

from .Optimizer import Optimizer
from ..CalcGraph.AbstractNode import CalcGraphNode
from ..DataPipelineTools.DataLoader import DataLoader


class SGDOptimizer(Optimizer):

    def __init__(self, learning_rate: float, max_batch_size: int, norm_grad: bool = False):
        self.__learning_rate__ = learning_rate
        self.__max_batch_size__ = max_batch_size
        self.__norm_grad__ = norm_grad

    def update_node_parameters(self, graph_node: CalcGraphNode, data: DataLoader) -> None:
        parameters_to_update = graph_node.get_parameters()
        data_to_calc_grads, labels = data.get_data_batch(self.__max_batch_size__)
        grad = np.squeeze(np.apply_along_axis(graph_node.calc_grads, -1, data_to_calc_grads))
        if self.__norm_grad__:
            grad /= np.linalg.norm(grad)
        predicted_values = np.squeeze(np.apply_along_axis(graph_node.calc_forward, -1, data_to_calc_grads))
        grad_directions = np.tile(np.sign(labels - predicted_values), (parameters_to_update.size ,1 )).T
        grad *= grad_directions
        grad = grad.mean(axis=0)
        parameters_to_update += self.__learning_rate__ * grad.mean(-1)
