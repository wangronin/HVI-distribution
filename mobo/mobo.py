import numpy as np

from .factory import init_from_config
from .surrogate_problem import SurrogateProblem
from .transformation import NonStandardTransform, SearchSpaceTransform
from .utils import Timer, calc_hypervolume, find_pareto_front

"""
Main algorithm framework for Multi-Objective Bayesian Optimization
"""


class MOBO:
    """Base class of algorithm framework, inherit this class with different configs to
    create new algorithm classes
    """

    config = {}

    def __init__(self, problem, n_iter, ref_point, framework_args):
        """
        Input:
            problem: the original / real optimization problem
            n_iter: number of iterations to optimize
            ref_point: reference point for hypervolume calculation
            framework_args: arguments to initialize each component of the framework
        """
        self.real_problem = problem
        self.n_var, self.n_obj = problem.n_var, problem.n_obj
        self.n_iter = n_iter
        self.ref_point = ref_point

        if hasattr(problem, "search_space"):
            self.search_space = problem.search_space
            self.transformation = SearchSpaceTransform(self.search_space)
        else:
            self.transformation = NonStandardTransform()

        # framework components
        framework_args["surrogate"]["n_var"] = self.n_var  # for surrogate fitting
        framework_args["surrogate"]["n_obj"] = self.n_obj  # for surroagte fitting
        framework_args["solver"]["n_obj"] = self.n_obj  # for MOEA/D-EGO
        framework = init_from_config(self.config, framework_args)

        self.surrogate_model = framework["surrogate"]  # surrogate model
        self.acquisition = framework["acquisition"]  # acquisition function
        self.solver = framework["solver"]  # multi-objective solver for finding the paretofront
        if self.solver.__class__.__name__ == "CMAESSolver":
            del self.solver.algo_kwargs["solver"]
            # del self.algo_kwargs['pop_size']
            del self.solver.algo_kwargs["n_cell"]
            del self.solver.algo_kwargs["n_process"]
            del self.solver.algo_kwargs["cell_size"]
            del self.solver.algo_kwargs["buffer_origin"]
            del self.solver.algo_kwargs["buffer_origin_constant"]
            del self.solver.algo_kwargs["delta_b"]
            del self.solver.algo_kwargs["label_cost"]
            del self.solver.algo_kwargs["delta_p"]
            del self.solver.algo_kwargs["delta_s"]
            del self.solver.algo_kwargs["n_grid_sample"]
            del self.solver.algo_kwargs["n_obj"]
            self.solver.algo_kwargs['sigma'] = self.solver.algo_kwargs['sigma'] * np.min(problem.xu - problem.xl)
            # del self.solver.algo_kwargs["pop_size"]
            # self.solver.algo_kwargs["n-gen"] = \
            #     self.solver.algo_kwargs["n-gen"] * self.solver.algo_kwargs["restarts"]

        self.selection = framework[
            "selection"
        ]  # selection method for choosing new (batch of) samples to evaluate on real problem

        # to keep track of data and pareto information (current status of algorithm)
        self.X = None
        self.Y = None
        self.sample_num = 0
        self.status = {
            "pset": None,
            "pfront": None,
            "hv": None,
            "ref_point": self.ref_point,
        }
        # other component-specific information that needs to be stored or exported
        self.info = None

    def _update_status(self, X, Y):
        """
        Update the status of algorithm from data
        """
        if self.sample_num == 0:
            self.X = X
            self.Y = Y
        else:
            self.X = np.vstack([self.X, X])
            self.Y = np.vstack([self.Y, Y])
        self.sample_num += len(X)

        self.status["pfront"], pfront_idx = find_pareto_front(self.Y, return_index=True)
        self.status["pset"] = self.X[pfront_idx]
        self.status["hv"] = calc_hypervolume(self.status["pfront"], self.ref_point)

    def solve(self, X_init, Y_init):
        """
        Solve the real multi-objective problem from initial data (X_init, Y_init)
        """
        # determine reference point from data if not specified by arguments
        if self.ref_point is None:
            self.ref_point = np.max(Y_init, axis=0)

        self.selection.set_ref_point(self.ref_point)
        self._update_status(X_init, Y_init)
        self.acquisition.n0 = len(X_init)
        global_timer = Timer()

        for i in range(self.n_iter):
            print("========== Iteration %d ==========" % i)
            timer = Timer()

            # data normalization
            self.transformation.fit(self.X, self.Y)
            X, Y = self.transformation.do(self.X, self.Y)

            # build surrogate models
            self.surrogate_model.fit(X, Y)
            timer.log("Surrogate model fitted")

            # create acquisition functions
            self.acquisition.fit(X, Y)

            # solve surrogate problem
            surr_problem = SurrogateProblem(
                self.real_problem,
                self.surrogate_model,
                self.acquisition,
                self.transformation,
            )
                
            acquisition_func = type(self.acquisition).__name__ 
            if acquisition_func in ['UCB'] or acquisition_func.startswith('HVI_UCB'):
                surr_problem.n_obj = 1
                
            # if type(self.acquisition).__name__ in ("HVI_UCB", "UCB"):
            #     surr_problem.n_obj = 1

            solution = self.solver.solve(surr_problem, X, Y)
            timer.log("Surrogate problem solved")

            # batch point selection
            self.selection.fit(X, Y)
            X_next, self.info = self.selection.select(
                solution, self.surrogate_model, self.status, self.transformation
            )
            # taking the precision into account
            if hasattr(self, "search_space"):
                X_next = self.search_space.round(self.transformation.undo(X_next))
            timer.log("Next sample batch selected")

            # update dataset
            Y_next = self.real_problem.evaluate(X_next)
            self._update_status(X_next, Y_next)
            timer.log("New samples evaluated")

            # statistics
            global_timer.log("Total runtime", reset=False)
            print(f"Total evaluations: {self.sample_num}, hypervolume: {self.status['hv']:.4f}\n")

            # return new data iteration by iteration
            yield X_next, Y_next

    def __str__(self):
        return (
            "========== Framework Description ==========\n"
            + f"# algorithm: {self.__class__.__name__}\n"
            + f"# surrogate: {self.surrogate_model.__class__.__name__}\n"
            + f"# acquisition: {self.acquisition.__class__.__name__}\n"
            + f"# solver: {self.solver.__class__.__name__}\n"
            + f"# selection: {self.selection.__class__.__name__}\n"
        )
