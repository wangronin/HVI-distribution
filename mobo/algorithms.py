from .mobo import MOBO

"""
High-level algorithm specifications by providing config
"""


class DGEMO(MOBO):
    """
    DGEMO
    """

    config = {
        "surrogate": "gp",
        "acquisition": "identity",
        "solver": "discovery",
        "selection": "dgemo",
    }


class TSEMO(MOBO):
    """
    TSEMO
    """

    config = {
        "surrogate": "ts",
        "acquisition": "identity",
        "solver": "nsga2",
        "selection": "hvi",
    }


class USEMO_EI(MOBO):
    """
    USeMO, using EI as acquisition
    """

    config = {
        "surrogate": "gp",
        "acquisition": "ei",
        "solver": "nsga2",
        "selection": "hvi",
    }


class MOEAD_EGO(MOBO):
    """
    MOEA/D-EGO
    """

    config = {
        "surrogate": "gp",
        "acquisition": "ei",
        "solver": "moead",
        "selection": "moead",
    }


class ParEGO(MOBO):
    """
    ParEGO
    """

    config = {
        "surrogate": "gp",
        "acquisition": "ei",
        "solver": "parego",
        "selection": "random",
    }


"""
Define new algorithms here
"""


class Custom(MOBO):
    """
    Totally rely on user arguments to specify each component
    """

    config = None


class HVIC_M1_ES(MOBO):
    """
    HVI_UCB_M1
    """

    config = {
        "surrogate": "gp",
        "acquisition": "hvi_ucb_m1",
        "solver": "cmaes",  # minimize abs(beta - ci)
        "selection": "MinCriterion",
    }
    
class HVIC_M2_ES(MOBO):
    """
    HVI_UCB_m2
    """

    config = {
        "surrogate": "gp",
        "acquisition": "hvi_ucb_m2", # max a, get -a value 
        "solver": "cmaes",  
        "selection": "MinCriterion", 
    }
    
class HVIC_M3_ES(MOBO):
    """
    HVI_UCB_m3
    """

    config = {
        "surrogate": "gp",
        "acquisition": "hvi_ucb_m1", # minimize abs(beta - ci)
        "solver": "cmaes", 
        "selection": "MaxCriterion", # select x with maximal a-value and minimal abs(beta-ci)
    }

class HVIC_M4_ES(MOBO):
    """
    HVI_UCB_m4
    """

    config = {
        "surrogate": "gp",
        "acquisition": "hvi_ucb_m3", # max hvic in the non-dominated space
        "solver": "cmaes", 
        "selection": "MinCriterion", # 
    }
    
class HVIC_M31_ES(MOBO):
    """
    HVI_UCB_m31
    """

    config = {
        "surrogate": "gp",
        "acquisition": "hvi_ucb_m1", # minimize abs(beta - ci)
        "solver": "cmaes", 
        "selection": "MinCriterionA", # select x with maximal a-value and minimal abs(beta-ci)
    }
    
    
    
    
    
    
class HVIC_M1_GA(MOBO):
    """
    HVI_UCB_m1
    """

    config = {
        # "surrogate": "gp",
        # "acquisition": "hvi_ucb_m1",
        # "solver": "ga",
        # "selection": "HVI_UCB_Uncertainty",
    }
    
class HVIC_M2_GA(MOBO):
    """
    HVI_UCB_m2
    """

    config = {
        # "surrogate": "gp",
        # "acquisition": "hvi_ucb_m2",
        # "solver": "ga",
        # "selection": "HVI_UCB_Uncertainty",
    }
    
class HVIC_M3_GA(MOBO):
    """
    HVI_UCB_m3
    """

    config = {
        # "surrogate": "gp",
        # "acquisition": "hvi_ucb_m3",
        # "solver": "ga",
        # "selection": "HVI_UCB_Uncertainty",
    }

class HVIC_M4_GA(MOBO):
    """
    HVI_UCB_m4
    """

    config = {
        # "surrogate": "gp",
        # "acquisition": "hvi_ucb_m4",
        # "solver": "ga",
        # "selection": "HVI_UCB_Uncertainty",
    }
    


class UCB(MOBO):
    """
    UCB
    """

    config = {
        "surrogate": "gp",
        "acquisition": "ucb",
        "solver": "cmaes",
        "selection": "hvi",
    }


def get_algorithm(name):
    """
    Get class of algorithm by name
    """
    algo = {
        "dgemo": DGEMO,
        "tsemo": TSEMO,
        "usemo-ei": USEMO_EI,
        "moead-ego": MOEAD_EGO,
        "parego": ParEGO,
        "custom": Custom,
        "hvic-m1-es": HVIC_M1_ES,
        "hvic-m2-es": HVIC_M2_ES,
        "hvic-m3-es": HVIC_M3_ES,
        "hvic-m31-es":HVIC_M31_ES,
        "hvic-m4-es": HVIC_M4_ES,
        "hvic-m1-ga": HVIC_M1_GA,
        "hvic-m2-ga": HVIC_M2_GA,
        "hvic-m3-ga": HVIC_M3_GA,
        "hvic-m4-ga": HVIC_M4_GA,
        "ucb": UCB,
    }
    return algo[name]
