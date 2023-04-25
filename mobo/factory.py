'''
Factory for importing different components of the MOBO framework by name
'''


def get_surrogate_model(name):
    from .surrogate_model import GaussianProcess, ThompsonSampling, RandomForest

    surrogate_model = {
        'gp': GaussianProcess,
        'ts': ThompsonSampling,
        'rf': RandomForest,
    }

    surrogate_model['default'] = GaussianProcess

    return surrogate_model[name]


def get_acquisition(name):
    from .acquisition import  Epsilon_PoI, UCB, Epsilon_PoI_Cut, PoHVI, NUCB
    acquisition = {
        # 'identity': IdentityFunc,
        # 'pi': PI,
        # 'ei': EI,
        'ucb': UCB,
        'nucb': NUCB,
        'pohvi': PoHVI,
        # 'hvi_ucb_m1': HVI_UCB_M1,
        # 'hvi_ucb_m2': HVI_UCB_M2,
        # 'hvi_ucb_m3': HVI_UCB_M3,
        # 'hvi_ucb_m4': HVI_UCB_M4,
        'epoi': Epsilon_PoI,
        'epoi_cut': Epsilon_PoI_Cut,
        # 'hvi_ucb_m3_epsilon': HVI_UCB_M3_EPSILON,
        # 'hvi_ucb_m3_epsilon_dr': HVI_UCB_M3_EPSILON_DR,

    }

    acquisition['default'] = UCB

    return acquisition[name]


def get_solver(name):
    from .solver import NSGA2Solver, MOEADSolver, ParetoDiscoverySolver
    # ParEGOSolver
    # from .solver import GASolver
    from .solver import GASolver
    from .solver import CMAESSolver

    solver = {
        'nsga2': NSGA2Solver,
        'moead': MOEADSolver,
        'discovery': ParetoDiscoverySolver,
        # 'parego': ParEGOSolver,
        'cmaes': CMAESSolver,
        'ga': GASolver,
    }

    solver['default'] = NSGA2Solver

    return solver[name]


def get_selection(name):
    from .selection import HVI, Uncertainty, Random, DGEMOSelect, MOEADSelect, HVI_UCB_Uncertainty, MaxCriterion, MinCriterion, MinCriterionA

    selection = {
        'hvi': HVI,
        'uncertainty': Uncertainty,
        'random': Random,
        'dgemo': DGEMOSelect,
        'moead': MOEADSelect,
        'HVI_UCB_Uncertainty': HVI_UCB_Uncertainty,
        'MaxCriterion': MaxCriterion,
        'MinCriterion': MinCriterion,
        'MinCriterionA': MinCriterionA,
    }

    selection['default'] = HVI

    return selection[name]


def init_from_config(config, framework_args):
    '''
    Initialize each component of the MOBO framework from config
    '''
    init_func = {
        'surrogate': get_surrogate_model,
        'acquisition': get_acquisition,
        'selection': get_selection,
        'solver': get_solver,
    }

    framework = {}
    for key, func in init_func.items():
        kwargs = framework_args[key]
        if config is None:
            # no config specified, initialize from user arguments
            name = kwargs[key]
        else:
            # initialize from config specifications, if certain keys are not provided, use default settings
            name = config[key] if key in config else 'default'
        framework[key] = func(name)(**kwargs)

    return framework
