from typing import Tuple

import numpy as np
from bayes_optim.search_space import Discrete, Integer, Real, SearchSpace

from arguments import get_args
from mobo.algorithms import get_algorithm
from utils import save_args, setup_logger
from visualization.data_export import DataExport

import autograd.numpy as anp


class NLPObjective:
    """Compute the objective values for tuning the hyperparameter of an NLP model"""

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed  # TODO: seed to used for initializing NLP models
        # TODO: the search space
        self.search_space = SearchSpace(
            [
                Real([-5, 5], "x1"),  # real-valued hyperparameter
                Real([-5, 5], "x2"),  # real-valued hyperparameter
                # Discrete(["A", "B", "C"], "x3"),  # discrete hyperparameter
                Integer([0, 10], "x4"),  # integer hyperparameter
            ]
        )
        
        # self.search 
        # the number of hyperparameters
        self.n_var = self.search_space.dim
        # the number of objectives
        self.n_obj = 2
        # the number of constraints
        self.n_constr = 0
        # lower bound
        self.xl= np.array([-5] * self.n_var)
        # upper bound 
        self.xu= np.array([5] * self.n_var)
            
        
        

        
        

    def sample(self, N: int) -> np.ndarray:
        """sample `N` point from the search space randomly"""
        return self.search_space.sample(N, method="LHS")

    def evaluate(self, hyperparameters: np.ndarray) -> np.ndarray:
        """evaluate the hyperparameters

        Parameters
        ----------
        hyperparameters : np.ndarray
            of shape (n_point, n_var)

        Returns
        -------
        np.ndarray
            of shape (n_point, n_obj)
        """
        values: list = []
        for par in hyperparameters:  # NOTE: this can be parallelized
            values.append(self._evaluate(par))
        return np.column_stack(list(zip(*values)))

    def _evaluate(self, par: np.ndarray, **kwargs) -> Tuple[float, float]:
        # TODO:
        # we should parameterize an NLP model using `par` and compute the target metrics
        # on some data set, e.g., for pytorch models
        # model.set_params(par)
        # model.fit(dataloader_train)
        # metric1, metric2 = [], []
        # with torch.no_grad():
        #   model.eval()
        #   for data, target in dataloader_val:
        #       output = model.predict(data)
        #       blue2 <- target, output
        #       blue4 <- target, output
        #       metric1.append(blue2.cpu().numpy())
        #       metric2.append(blue4.cpu().numpy())
        # return np.mean(metric1), np.mean(metric2) # or compute the median
        
        f1 = np.random.randn()
        f2 = np.random.randn()
        # out["F"] = anp.column_stack([f1, f2])

        return np.random.randn(), np.random.randn()  # mockup values to be replaced
    
    # def _evaluate(self, x, out, *args, requires_F=True, **kwargs):
    #     if requires_F:
    #         f1 = np.random.randn()
    #         f2 = np.random.randn()

    #         out["F"] = anp.column_stack([f1, f2])


def experiment():
    # load arguments
    args, framework_args = get_args()

    # set seed
    np.random.seed(args.seed)
    problem = NLPObjective(args.seed)

    # initialize optimizer
    optimizer = get_algorithm(args.algo)(problem, args.n_iter, args.ref_point, framework_args)

    # save arguments & setup logger
    save_args(args, framework_args)
    logger = setup_logger(args)

    X_init = problem.sample(10)  # take a small set of initial guess
    Y_init = problem.evaluate(X_init)

    # initialize data exporter
    exporter = DataExport(optimizer, X_init, Y_init, args)

    # optimization
    solution = optimizer.solve(X_init, Y_init)

    for _ in range(args.n_iter):
        # get new design samples and corresponding performance
        X_next, Y_next = next(solution)

        # update & export current status to csv
        exporter.update(X_next, Y_next)
        exporter.write_csvs()

    # close logger
    if logger is not None:
        logger.close()


if __name__ == "__main__":
    # run this script with command
    # python example --algo ucb hvi_ucb --batch-size 10 --n-iter 500 --seed 42"
    experiment()
