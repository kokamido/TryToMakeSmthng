from abc import ABC, abstractmethod

from ..CalcGraph.AbstractGraph import CalcGraph
from ..DataPipelineTools.DataLoader import DataLoader
from ..Losses.AbstractLoss import AbstractLoss


class Optimizer(ABC):
    @abstractmethod
    def update_node_parameters(
        self, graph_node: CalcGraph, loss: AbstractLoss, data_loader: DataLoader
    ) -> None:
        pass
