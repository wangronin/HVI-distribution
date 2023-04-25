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


class NUCB(MOBO):
    """
    NUCB
    """

    config = {
        "surrogate": "gp",
        "acquisition": "nucb",
        "solver": "cmaes",
        "selection": "hvi",
    }

class POHVI(MOBO):
    """
    NUCB
    """

    config = {
        "surrogate": "gp",
        "acquisition": "pohvi",
        "solver": "cmaes",
        "selection": "hvi",
    }

class EPOI(MOBO):
    """
    NUCB
    """

    config = {
        "surrogate": "gp",
        "acquisition": "epoi",
        "solver": "cmaes",
        "selection": "hvi",
    }


def get_algorithm(name):
    """
    Get class of algorithm by name
    """
    algo = {
        # "dgemo": DGEMO,
        # "tsemo": TSEMO,
        # "usemo-ei": USEMO_EI,
        # "moead-ego": MOEAD_EGO,
        # "parego": ParEGO,
        # "custom": Custom,
        # "hvic-m1-es": HVIC_M1_ES,
        # "hvic-m2-es": HVIC_M2_ES,
        # "hvic-m3-es": HVIC_M3_ES,
        # "hvic-m31-es":HVIC_M31_ES,
        # "hvic-m4-es": HVIC_M4_ES,
        # "hvic-m1-ga": HVIC_M1_GA,
        # "hvic-m2-ga": HVIC_M2_GA,
        # "hvic-m3-ga": HVIC_M3_GA,
        # "hvic-m4-ga": HVIC_M4_GA,
        "ucb": UCB,
        "epoi": EPOI,
        "nucb": NUCB, 
        "pohvi": POHVI,


        # "pohvi": 
        # "epoi_c": EPOI_C,
        # "hvic-m4-es-e": HVIC_M4_ES_Epsilon,
        # "hvic-m4-es-e-dr": HVIC_M4_ES_Epsilon_DynamicRef,
    }
    return algo[name]
