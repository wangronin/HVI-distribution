from .solver import Solver
from pymoo.algorithms.so_cmaes import CMAES


class CMAESSolver(Solver):
    """
    Solver based on CMA-ES
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, algo=CMAES, **kwargs)
