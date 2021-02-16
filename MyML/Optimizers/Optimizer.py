from abc import ABC, abstractmethod
from ..CalcGraph.AbstractNode import CalcGraphNode
from ..DataPipelineTools.DataLoader import DataLoader


class Optimizer(ABC):

    @abstractmethod
    def update_node_parameters(self, graph_node: CalcGraphNode, data_loader: DataLoader) -> None:
        pass
