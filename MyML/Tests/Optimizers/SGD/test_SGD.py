import unittest

import numpy as np

from MyML.Algorithms.LinearRegression.LinearRegression import LinearRegression
from MyML.DataPipelineTools.DataLoader import GeneratorBasedLoader
from MyML.Helpers.TestHelpers.ConstantLoss import ConstantLoss
from MyML.Helpers.TestHelpers.SingleDirectionGradsStub import SingleDirectionGradsStub
from MyML.Losses.AbsoluteError import AbsoluteError
from MyML.Optimizers.SGD import SGDOptimizer


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
        np.testing.assert_array_almost_equal(parameters, [-1.0, -1.0, -1.0], 0.0001)

    def test_SGD_random_data_shapes(self):
        for _ in range(50):
            shape = []
            for _ in range(np.random.randint(low=1, high=8)):
                shape.append(np.random.randint(low=2, high=10))
            node = SingleDirectionGradsStub(shape=shape, grad_value=2, forward_value=2)
            loss = ConstantLoss(grads_value=1, loss_value=1)
            optimizer = SGDOptimizer(0.01, 1)
            data_loader = GeneratorBasedLoader(
                generator_func=lambda: (1, 1), calc_shapes_on_init=True
            )
            parameters = node.get_learnable_parameters()
            for _ in range(100):
                optimizer.update_node_parameters(node, loss, data_loader)
            delta = np.sum(np.abs(parameters - np.ones(shape=shape) * (-1)).ravel())
            self.assertAlmostEqual(delta, 0, delta=0.000001)

    def test_SGD_with_linear_regression_1d_data(self):
        for _ in range(10):
            dim = np.random.randint(low=1, high=25)
            weights = np.random.normal(size=dim)

            def get_data():
                point = np.random.normal(size=dim - 1)
                return point, np.dot(point, weights[:-1]) + weights[-1]

            reg = LinearRegression(([dim - 1]))
            loader = GeneratorBasedLoader(get_data)
            loss = AbsoluteError()
            opt = SGDOptimizer(0.001, 10)
            params = reg.get_learnable_parameters()
            iter_num = 0
            while np.sum(np.abs(weights - params)) > 0.01:
                iter_num += 1
                if iter_num == 20000:
                    self.assertTrue(
                        False,
                        "Linear regression with SGD takes too long time to converge",
                    )
                opt.update_node_parameters(reg, loss, loader)
            self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
