from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np
from sympy import symbols

x, xl_, yl_, C_ = symbols("x xl_ yl_ C_")
curve_2d = C_ / (x - xl_) + yl_


def _set_cells(pareto_front: np.ndarray, reference_point: np.ndarray) -> Dict[Tuple[int, int], dict]:
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


class HypervolumeLevelCurve:
    def __init__(self, pareto_front: np.ndarray, reference_point: np.ndarray) -> None:
        # sorting in the increasing order of the first component
        self.pareto_front = pareto_front[pareto_front[:, 0].argsort()]
        self.reference_point = reference_point
        self.cell_info = _set_cells(self.pareto_front, self.reference_point)
        self.N = len(pareto_front)

    def compute(self, level: float) -> List[List[float, float, Callable]]:
        # always start with cell (0, 0)
        info = self.cell_info[(0, 0)]
        x0, x1 = info["x0"], info["xmax"]
        xl, yl = info["xl"], info["yl"]
        level_curve = [[x0, x1, curve_2d.subs({xl_: xl, yl_: yl, C_: level})]]
        i, j = 0, 1
        while i < self.N or j < self.N:
            info = self.cell_info[(i, j)]
            xmax, ymin = info["xmax"], info["ymin"]
            xl, yl = info["xl"], info["yl"]
            if i == 0 and j == 1:
                x0 = info["x0"]
                y0 = level / (x0 - xl) + ymin
            C = (x0 - xl) * (y0 - yl)
            x1 = min(C / (ymin - yl) + xl, xmax)  # the last value of `x`
            level_curve.append([x0, x1, curve_2d.subs({xl_: xl, yl_: yl, C_: C})])
            x0 = x1  # the first value of `x`
            y0 = ymin if x1 < xmax else C / (xmax - xl) + yl  # the first value of `y`
            if x1 < xmax:
                i += 1
            else:
                j += 1

        # always end with cell (`N`, `N`)
        info = self.cell_info[(i, j)]
        x1, xl, yl = info["xmax"], info["xl"], info["yl"]
        level_curve.append([x0, x1, curve_2d.subs({xl_: xl, yl_: yl, C_: level})])
        return level_curve
