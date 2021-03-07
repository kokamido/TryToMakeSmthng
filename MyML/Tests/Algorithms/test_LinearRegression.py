import unittest

import numpy as np

from MyML.Algorithms.LinearRegression.LinearRegression import LinearRegression
from MyML.DataPipelineTools.DataLoader import GeneratorBasedLoader
from MyML.Losses.AbsoluteError import AbsoluteError
from MyML.Optimizers.BatchGradientOptimizer import BatchGradientOptimizer


class SGDTests(unittest.TestCase):
    def test_linear_regression_BatchGradientOptimizer_with(self):
        for _ in range(10):
            dim = np.random.randint(low=1, high=25)
            weights = np.random.normal(size=dim)

            def get_data():
                point = np.random.normal(size=dim - 1)
                return point, np.dot(point, weights[:-1]) + weights[-1]

            reg = LinearRegression(([dim - 1]))
            loader = GeneratorBasedLoader(get_data)
            loss = AbsoluteError()
            opt = BatchGradientOptimizer(0.001, 25)
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
