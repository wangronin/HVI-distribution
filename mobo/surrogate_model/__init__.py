import numpy as np
from bayes_optim.surrogate import RandomForest as _RandomForest

from .gaussian_process import GaussianProcess
from .thompson_sampling import ThompsonSampling


class RandomForest(_RandomForest):
    def evaluate(self, X, std=False, calc_gradient=False, calc_hessian=False):
        F, MSE = self.predict(X, eval_MSE=True)
        S = np.sqrt(MSE)
        return {"F": F, "S": S}


__all__ = ["RandomForest"]
