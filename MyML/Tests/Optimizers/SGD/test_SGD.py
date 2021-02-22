import unittest
import numpy as np

from MyML.DataPipelineTools.DataLoader import GeneratorBasedLoader
from MyML.Optimizers.SGD import SGDOptimizer
from MyML.Helpers.TestHelpers.ConstantLoss import ConstantLoss
from MyML.Helpers.TestHelpers.SingleDirectionGradsStub import (
    SingleDirectionGradsStub,
)


class SGDTests(unittest.TestCase):
    @staticmethod
    def test_SGD_most_simple_case():
        node = SingleDirectionGradsStub(shape=[3], grad_value=2, forward_value=2)
        loss = ConstantLoss(grads_value=1, loss_value=1)
        optimizer = SGDOptimizer(0.01, 1)
        data_loader = GeneratorBasedLoader(generator_func=lambda: (1, 1))
        parameters = node.get_learnable_parameters()
        for _ in range(100):
            optimizer.update_node_parameters(node, loss, data_loader)
        np.testing.assert_array_almost_equal(parameters, [-12.0, -1.0, -1.0], 0.0001)


if __name__ == "__main__":
    unittest.main()
