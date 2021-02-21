# mypy: ignore-errors


import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_regression

from MyML.Algorithms.LinearRegression.LinearRegression import LinearRegression
from MyML.DataPipelineTools.DataLoader import GeneratorBasedLoader
from MyML.Losses.AbsoluteError import AbsoluteError
from MyML.Optimizers.SGD import SGDOptimizer

dim = 3
weights = np.random.normal(size=dim)

def get_data():
    point = np.random.normal(size=dim-1)
    return point, np.dot(point, weights[:-1])+weights[-1]

reg = LinearRegression(([dim-1]))
loader = GeneratorBasedLoader(get_data)
loss = AbsoluteError()
opt = SGDOptimizer(0.001, 10)
params = reg.get_learnable_parameters()
while np.sum(np.abs(weights - params)) > 0.01:
    opt.update_node_parameters(reg, loss, loader)
print(reg.get_learnable_parameters())
print(weights)
plt.show()
