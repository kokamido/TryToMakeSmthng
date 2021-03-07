import unittest

import numpy as np

from MyML.DataPipelineTools.DataLoader import GeneratorBasedLoader, StubDataLoader
from MyML.Helpers.TestHelpers.ConstantLoss import ConstantLoss
from MyML.Helpers.TestHelpers.OptimizationTestFunc import BealeFunc, SphereFunc
from MyML.Helpers.TestHelpers.SingleDirectionGradsStub import SingleDirectionGradsStub
from MyML.Losses.IdenticalLoss import IdenticalLoss
from MyML.Optimizers.BatchGradientOptimizer import BatchGradientOptimizer


class BatchGradientOptimizerTests(unittest.TestCase):
    @staticmethod
    def test_BatchGradientOptimizer_most_simple_case():
        node = SingleDirectionGradsStub(shape=[3], grad_value=2, forward_value=2)
        loss = ConstantLoss(grads_value=1, loss_value=1)
        optimizer = BatchGradientOptimizer(0.01, 1)
        data_loader = GeneratorBasedLoader(generator_func=lambda: (1, 1))
        parameters = node.get_learnable_parameters()
        for _ in range(100):
            optimizer.update_node_parameters(node, loss, data_loader)
        np.testing.assert_array_almost_equal(parameters, [-1.0, -1.0, -1.0], 0.0001)

    @staticmethod
    def test_BatchGradientOptimizer_most_simple_case_big_batch():
        node = SingleDirectionGradsStub(shape=[3], grad_value=2, forward_value=2)
        loss = ConstantLoss(grads_value=1, loss_value=1)
        optimizer = BatchGradientOptimizer(0.01, 100)
        data_loader = GeneratorBasedLoader(generator_func=lambda: (1, 1))
        parameters = node.get_learnable_parameters()
        for _ in range(100):
            optimizer.update_node_parameters(node, loss, data_loader)
        np.testing.assert_array_almost_equal(parameters, [-1.0, -1.0, -1.0], 0.0001)

    def test_BatchGradientOptimizer_random_data_shapes_batch_size_1(self):
        for _ in range(50):
            shape = []
            for _ in range(np.random.randint(low=1, high=8)):
                shape.append(np.random.randint(low=2, high=10))
            node = SingleDirectionGradsStub(shape=shape, grad_value=2, forward_value=2)
            loss = ConstantLoss(grads_value=1, loss_value=1)
            optimizer = BatchGradientOptimizer(0.01, 1)
            data_loader = GeneratorBasedLoader(generator_func=lambda: (1, 1))
            parameters = node.get_learnable_parameters()
            for _ in range(100):
                optimizer.update_node_parameters(node, loss, data_loader)
            delta = np.sum(np.abs(parameters - np.ones(shape=shape) * (-1)).ravel())
            self.assertAlmostEqual(delta, 0, delta=1e-6)

    def test_BatchGradientOptimizer_random_data_shapes_batch_size_10(self):
        for _ in range(50):
            shape = []
            for _ in range(np.random.randint(low=1, high=8)):
                shape.append(np.random.randint(low=2, high=10))
            node = SingleDirectionGradsStub(shape=shape, grad_value=2, forward_value=2)
            loss = ConstantLoss(grads_value=1, loss_value=1)
            optimizer = BatchGradientOptimizer(0.01, 10)
            data_loader = GeneratorBasedLoader(generator_func=lambda: ([1], [1]))
            parameters = node.get_learnable_parameters()
            for _ in range(100):
                optimizer.update_node_parameters(node, loss, data_loader)
            delta = np.sum(np.abs(parameters - np.ones(shape=shape) * (-1)).ravel())
            self.assertAlmostEqual(delta, 0, delta=1e-6)

    def test_BatchGradientOptimizer_sphere_optimization(self):
        for _ in range(10):
            start_point = np.random.uniform(low=1.0, high=1.0, size=10)
            end_point = np.zeros_like(start_point)
            sphere_func = SphereFunc(start_point)
            optimizer = BatchGradientOptimizer(0.25, 1)
            loss = IdenticalLoss()
            data_loader = StubDataLoader()
            point = sphere_func.get_learnable_parameters()
            iter_num = 0
            while np.sum(np.abs(point - end_point)) > 1e-2:
                iter_num += 1
                if iter_num == 20000:
                    self.assertTrue(
                        False,
                        "Sphere minimum search with SGD takes too long time to converge",
                    )
                optimizer.update_node_parameters(sphere_func, loss, data_loader)
                sphere_func.point = point
            self.assertTrue(True)

    def test_BatchGradientOptimizer_Beale_func_optimization(self):
        for _ in range(10):
            noise = np.random.uniform(low=-0.75, high=0.75, size=2)
            start_point = np.array([3.0, 0.5], dtype=np.float128) + noise
            end_point = np.array([3.0, 0.5], dtype=np.float128)
            func = BealeFunc(start_point)
            optimizer = BatchGradientOptimizer(0.01, 1)
            loss = IdenticalLoss()
            data_loader = StubDataLoader()
            point = func.get_learnable_parameters()
            iter_num = 0
            while np.sum(np.abs(point - end_point)) > 1e-2:
                iter_num += 1
                if iter_num == 50000:
                    self.assertTrue(
                        False,
                        "Beale function optimization with SGD takes too long time to converge",
                    )
                optimizer.update_node_parameters(func, loss, data_loader)
                func.point = point
            self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
