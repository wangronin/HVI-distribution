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
        "epoi": EPOI,
        "pohvi": POHVI,
    }
    return algo[name]
