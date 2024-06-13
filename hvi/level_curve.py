from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from sympy import Expr, symbols

x, xl_, yl_, C_ = symbols("x xl_ yl_ C_")
curve_2d = C_ / (x - xl_) + yl_


def _set_cells(
    pareto_front: np.ndarray, reference_point: np.ndarray
) -> Dict[Tuple[int, int], Dict[str, float]]:
    cell_info = dict()
    pareto_front_ = np.vstack(([reference_point[0], np.inf], pareto_front, [np.inf, reference_point[1]]))
    N = len(pareto_front_)
    for i in range(N - 1):  # `i` indexes the row/on y-axis
        for j in range(i, N - 1):  # `j` indexes the column/on x-axis
            cell_info[(i, j)] = dict(
                x0=pareto_front_[j, 0],
                ymin=pareto_front_[i + 1, 1],
                xmax=pareto_front_[j + 1, 0],
                xl=pareto_front_[i, 0],
                yl=pareto_front_[j + 1, 1],
            )
    return cell_info


class HypervolumeImprovementLevelCurve2D:
    def __init__(self, pareto_front: np.ndarray, reference_point: np.ndarray) -> None:
        """Compute the level curve of hypervolume improvement in 2D. Maximization is assumed.

        Args:
            pareto_front (np.ndarray): the approximation set to the Pareto front, of shape (`N`, 2)
            reference_point (np.ndarray): the reference point, of shape (2, )
        """
        # sorting in the increasing order of the first component
        self.pareto_front = pareto_front[pareto_front[:, 0].argsort()]
        self.reference_point = reference_point
        self.cell_info = _set_cells(self.pareto_front, self.reference_point)
        self.N = len(pareto_front)

    def compute(self, level: float) -> List[List[float, float, Expr]]:
        """HVI's level curve at `level`. It is a piecewise hyperbola.

        Args:
            level (float): the hypervolume improvement value

        Returns:
            List[List[float, float, Expr]]: list of hyperbola pieces. Each piece is specified
            by (x_start, x_end, y = f(x)), where the parametric function f(x) is implemented
            as `sympy`'s expression.
        """
        level_curve = []
        i, j = 0, 0  # always start with cell (0, 0)
        while i <= self.N and j <= self.N:  # always end with cell (`N`, `N`)
            info = self.cell_info[(i, j)]
            xmax, ymin, xl, yl = info["xmax"], info["ymin"], info["xl"], info["yl"]
            if i == 0 and j == 0:
                x0, x1, C = info["x0"], xmax, level
            else:
                C = (x0 - xl) * (y0 - yl)
                x1 = min(C / (ymin - yl) + xl, xmax)  # the last value of `x`
            level_curve.append([x0, x1, curve_2d.subs({xl_: xl, yl_: yl, C_: C})])
            x0 = x1  # the initial value of `x`
            y0 = ymin if x1 < xmax else C / (xmax - xl) + yl  # the initial value of `y`
            i += int(x1 < xmax)
            j += int(x1 == xmax)
        return level_curve
