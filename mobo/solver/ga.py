from .solver import Solver
from pymoo.algorithms.so_genetic_algorithm import GA



class GASolver(Solver):
    '''
    Solver based on GA
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, algo=GA, **kwargs)
