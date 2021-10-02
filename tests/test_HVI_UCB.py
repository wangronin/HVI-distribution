import numpy as np
from mobo.acquisition import HVI_UCB

X = Y = np.random.rand(500, 2)
ac = HVI_UCB().fit(X, Y)

ac.evaluate({"F": np.array([[0.0, 0.1]] * 10), "S": np.array([[0.1, 0.5]] * 10)})
