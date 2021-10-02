import numpy as np
from mobo.acquisition import HVI_UCB

X = Y = np.array(
    [
        [0, 20],
        [1, 19],
        [2, 18],
        [3, 17],
        [4, 16],
        [5, 15],
        [6, 14],
        [7, 13],
        [8, 12],
        [9, 11],
        [10, 10],
        [11, 9],
        [12, 8],
        [13, 7],
        [14, 6],
        [15, 5],
        [16, 4],
        [17, 3],
        [18, 2],
        [19, 1],
        [20, 0],
    ]
)
ac = HVI_UCB().fit(X, Y)
ac.evaluate({"F": np.array([[5, 0.1]] * 10), "S": np.array([[1, 0.5]] * 10)})
