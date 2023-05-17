import os
import sys
from time import time

import numpy as np
import yaml
from pymoo.factory import get_performance_indicator


def get_result_dir(args):
    """
    Get directory of result location (result/problem/subfolder/algo/seed/)
    """
    top_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")
    exp_name = "" if args.exp_name is None else "-" + args.exp_name
    algo_name = args.algo + exp_name
    result_dir = os.path.join(top_dir, args.problem, args.subfolder, algo_name, str(args.seed))
    os.makedirs(result_dir, exist_ok=True)
    return result_dir


def save_args(general_args, framework_args):
    """
    Save arguments to yaml file
    """
    all_args = {"general": vars(general_args)}
    all_args.update(framework_args)

    result_dir = get_result_dir(general_args)
    args_path = os.path.join(result_dir, "args.yml")

    os.makedirs(os.path.dirname(args_path), exist_ok=True)
    with open(args_path, "w") as f:
        yaml.dump(all_args, f, default_flow_style=False, sort_keys=False)


def setup_logger(args):
    """
    Log to file if needed
    """
    logger = None

    if args.log_to_file:
        result_dir = get_result_dir(args)
        log_path = os.path.join(result_dir, "log.txt")
        logger = open(log_path, "w")
        sys.stdout = logger

    return logger


class Timer:
    """
    For time recording and message logging
    """

    def __init__(self):
        self.t = time()

    def log(self, string=None, reset=True):
        msg = "%.2fs" % (time() - self.t)

        if string is not None:
            msg = string + ": " + msg
        print(msg)

        if reset:
            self.t = time()

    def reset(self):
        self.t = time()


def find_pareto_front(Y, return_index=False):
    """
    Find pareto front (undominated part) of the input performance data.
    """
    if len(Y) == 0:
        return np.array([])
    sorted_indices = np.argsort(Y.T[0])
    pareto_indices = []
    for idx in sorted_indices:
        # check domination relationship
        a = np.all(Y <= Y[idx], axis=1)
        b = np.any(Y < Y[idx], axis=1)
        if not np.any(np.logical_and(a, b)):
            pareto_indices.append(idx)
    pareto_front = Y[pareto_indices].copy()

    if return_index:
        return pareto_front, pareto_indices
    else:
        return pareto_front


def calc_hypervolume(pfront, ref_point):
    """
    Calculate hypervolume of pfront based on ref_point
    """
    hv = get_performance_indicator("hv", ref_point=ref_point)
    return hv.calc(pfront)


def safe_divide(x1, x2):
    """
    Divide x1 / x2, return 0 where x2 == 0
    """
    return np.divide(x1, x2, out=np.zeros(np.broadcast(x1, x2).shape), where=(x2 != 0))


def expand(x, axis=-1):
    """
    Concise way of expand_dims
    """
    return np.expand_dims(x, axis=axis)
