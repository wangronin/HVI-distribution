import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from abc import ABC, abstractmethod
import numpy as np
from sklearn.cluster import KMeans
from pymoo.factory import get_performance_indicator
from pymoo.algorithms.nsga2 import calc_crowding_distance

from scipy.optimize import root_scalar, minimize
from scipy import optimize
from scipy.optimize import root
from .hv_improvement import HypervolumeImprovement

"""
Selection methods for new batch of samples to evaluate on real problem
"""


class Selection(ABC):
    """
    Base class of selection method
    """

    def __init__(self, batch_size, ref_point=None, **kwargs):
        self.batch_size = batch_size
        self.ref_point = ref_point

    def fit(self, X, Y):
        """
        Fit the parameters of selection method from data
        """
        pass

    def set_ref_point(self, ref_point):
        self.ref_point = ref_point

    @abstractmethod
    def select(self, solution, surrogate_model, status, transformation):
        """
        Select new samples from solution obtained by solver
        Input:
            solution['x']: design variables of solution
            solution['y']: acquisition values of solution
            solution['algo']: solver algorithm, having some relevant information from optimization
            surrogate_model: fitted surrogate model
            status['pset']: current pareto set found
            status['pfront]: current pareto front found
            status['hv']: current hypervolume
            transformation: data normalization for surrogate model fitting
            (some inputs may not be necessary for some selection criterion)
        Output:
            X_next: next batch of samples selected
            info: other informations need to be stored or exported, None if not necessary
        """
        pass


class HVI(Selection):
    """
    Hypervolume Improvement
    """

    def select(self, solution, surrogate_model, status, transformation):

        pred_pset = solution["x"]
        val = surrogate_model.evaluate(pred_pset)
        pred_pfront = val["F"]
        pred_pset, pred_pfront = transformation.undo(pred_pset, pred_pfront)

        curr_pfront = status["pfront"].copy()
        hv = get_performance_indicator("hv", ref_point=self.ref_point)
        idx_choices = np.ma.array(
            np.arange(len(pred_pset)), mask=False
        )  # mask array for index choices
        next_batch_indices = []

        # greedily select indices that maximize hypervolume contribution
        for _ in range(self.batch_size):
            curr_hv = hv.calc(curr_pfront)
            max_hv_contrib = 0.0
            max_hv_idx = -1
            for idx in idx_choices.compressed():
                # calculate hypervolume contribution
                new_hv = hv.calc(np.vstack([curr_pfront, pred_pfront[idx]]))
                hv_contrib = new_hv - curr_hv
                if hv_contrib > max_hv_contrib:
                    max_hv_contrib = hv_contrib
                    max_hv_idx = idx
            if (
                max_hv_idx == -1
            ):  # if all candidates have no hypervolume contribution, just randomly select one
                max_hv_idx = np.random.choice(idx_choices.compressed())

            idx_choices.mask[max_hv_idx] = True  # mask as selected
            curr_pfront = np.vstack(
                [curr_pfront, pred_pfront[max_hv_idx]]
            )  # add to current pareto front
            next_batch_indices.append(max_hv_idx)
        next_batch_indices = np.array(next_batch_indices)

        return pred_pset[next_batch_indices], None


class Uncertainty(Selection):
    """
    Uncertainty
    """

    def select(self, solution, surrogate_model, status, transformation):

        X = solution["x"]
        val = surrogate_model.evaluate(X, std=True)
        Y_std = val["S"]
        X = transformation.undo(x=X)

        uncertainty = np.prod(Y_std, axis=1)
        top_indices = np.argsort(uncertainty)[::-1][: self.batch_size]
        return X[top_indices], None


class Random(Selection):
    """
    Random selection
    """

    def select(self, solution, surrogate_model, status, transformation):
        X = solution["x"]
        X = transformation.undo(x=X)
        random_indices = np.random.choice(len(X), size=self.batch_size, replace=False)
        return X[random_indices], None


class DGEMOSelect(Selection):
    """
    Selection method for DGEMO algorithm
    """

    has_family = True

    def select(self, solution, surrogate_model, status, transformation):
        algo = solution["algo"]

        X_next, _, family_lbls_next = algo.propose_next_batch(
            status["pfront"], self.ref_point, self.batch_size, transformation
        )
        family_lbls, approx_pset, approx_pfront = algo.get_sparse_front(transformation)

        info = {
            "family_lbls_next": family_lbls_next,
            "family_lbls": family_lbls,
            "approx_pset": approx_pset,
            "approx_pfront": approx_pfront,
        }
        return X_next, info


class MOEADSelect(Selection):
    """
    Selection method for MOEA/D-EGO algorithm
    """

    def select(self, solution, surrogate_model, status, transformation):
        X, G, algo = solution["x"], solution["y"], solution["algo"]
        ref_dirs = algo.ref_dirs

        G_s = algo._decomposition.do(
            G, weights=ref_dirs, ideal_point=algo.ideal_point
        )  # scalarized acquisition value

        # build candidate pool Q
        Q_x, Q_dir, Q_g = [], [], []
        X_added = status["pset"].copy()
        for x, ref_dir, g in zip(X, ref_dirs, G_s):
            if (x != X_added).any(axis=1).all():
                Q_x.append(x)
                Q_dir.append(ref_dir)
                Q_g.append(g)
                X_added = np.vstack([X_added, x])
        Q_x, Q_dir, Q_g = np.array(Q_x), np.array(Q_dir), np.array(Q_g)

        batch_size = min(
            self.batch_size, len(Q_x)
        )  # in case Q is smaller than batch size

        if batch_size == 0:
            X_next = X[np.random.choice(len(X), self.batch_size, replace=False)]
            X_next = transformation.undo(x=X_next)
            return X_next, None

        # k-means clustering on X with weight vectors
        labels = KMeans(n_clusters=batch_size).fit_predict(
            np.column_stack([Q_x, Q_dir])
        )

        # select point in each cluster with lowest scalarized acquisition value
        X_next = []
        for i in range(batch_size):
            indices = np.where(labels == i)[0]
            top_idx = indices[np.argmin(Q_g[indices])]
            top_x = transformation.undo(x=Q_x[top_idx])
            X_next.append(top_x)
        X_next = np.array(X_next)

        # when Q is smaller than batch size
        if batch_size < self.batch_size:
            X_rest = X[
                np.random.choice(len(X), self.batch_size - batch_size, replace=False)
            ]
            X_next = np.vstack([X_next, transformation.undo(x=X_rest)])

        return X_next, None


class HVI_UCB_Uncertainty(Selection):
    """
    Uncertainty
    """

    def select(self, solution, surrogate_model, status, transformation):
        # version 1: repl
        #   select "x" of which a is smallest and hvi.cdf(a) == 0.95
        # X = solution['x']
        # val = surrogate_model.evaluate(X, std=True)

        # # kaifeng
        # # X = transformation.undo(x=X)

        # pf = status['pfront'].copy()
        # mu, sigma = val['F'], val['S']
        # rf = self.ref_point
        # n_candi = len(sigma)
        # hvi_ubc = np.array([0] *  n_candi)
        # for i in range(0,n_candi):
        #     hvi = HypervolumeImprovement(pf, rf, mu[i,:], sigma[i,:])
        #     func = lambda x: abs(hvi.cdf(x) - 0.95)
        #     # func = lambda x: abs(hvi.cdf_monte_carlo(x, n_sample=(1e5), eval_sd=False)- 0.95)
        #     # root = optimize.bisect(func, -999, 9e99, maxiter=500)
        #     # # sol = root(func, x0=0, args=(), method='hybr', jac=None, tol=None, callback=None, options=None)
        #     # hvi_ubc[i] = root

        #     sol = minimize(func, 1, method='CG', options={'maxiter': 30})
        #     hvi_ubc[i] = sol.x

        # top_indices = np.argsort(-1 * hvi_ubc)[::-1][:self.batch_size]

        # # uncertainty = np.prod(Y_std, axis=1)
        # # top_indices = np.argsort(uncertainty)[::-1][:self.batch_size]
        # return X[top_indices], None

        # version 2: same with Version 1, but is faster as it directly uses the previous calculations
        X = solution["x"]
        # solution['a'][solution['y'] == np.inf] =
        top_indices = np.argsort(-1 * solution["a"], axis=0)[::-1][: self.batch_size][
            0
        ]  # min
        return X[top_indices], None

        # version 3: find the x of which UBC is max
        # X = solution['x']
        # Y = solution['y']
        # top_indices = np.argsort(-1 * Y)[::-1][:self.batch_size]
        # if self.batch_size == 1:
        #     print('UBC = %f' %Y[top_indices[0]])
        #     return X[top_indices[0]], None
        # else:
        #     return X[top_indices], None
