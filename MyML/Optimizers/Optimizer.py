from abc import ABC, abstractmethod

from MyML.CalcGraph.AbstractGraph import CalcGraph
from MyML.DataPipelineTools.DataLoader import DataLoader
from MyML.Losses.AbstractLoss import AbstractLoss


class Optimizer(ABC):
    @abstractmethod
    def update_node_parameters(
        self, graph_node: CalcGraph, loss: AbstractLoss, data_loader: DataLoader
    ) -> None:
        pass
