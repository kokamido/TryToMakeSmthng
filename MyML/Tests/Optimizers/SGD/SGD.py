import unittest
from MyML.Optimizers.SGD import SGDOptimizer
from MyML.Tests.Optimizers.Helpers.SingleDirectionGradsStub import SingleDirectionGradsStub

class MyTestCase(unittest.TestCase):

    def test_it_just_works(self):
        node = SingleDirectionGradsStub(shape=[3])
        optimizer = SGDOptimizer(0.01, 1)
        parameters = node.get_learnable_parameters()
        for _ in range(100):
            optimizer.update_node_parameters(node,)
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
