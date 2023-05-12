import os

os.environ["OMP_NUM_THREADS"] = "1"  # speed up
import matplotlib.pyplot as plt
import numpy as np

from arguments import get_args
from mobo.algorithms import get_algorithm
from mobo.hv_improvement import HypervolumeImprovement
from mobo.utils import find_pareto_front
from problems.common import build_problem
from utils import save_args, setup_logger
from visualization.data_export import DataExport

# plt.style.use("ggplot")

seed = 42
n_decision_vars = 2
n_objectives = 2
n_init_samples = 20


def plot_attainment_boundary(approximation, ref, ax):
    X = approximation[approximation[:, 0].argsort()]
    X = np.vstack([[X[0, 0], ref[1]], X, [ref[0], X[-1, 1]]])
    v_pos = X[1:-1, 0]
    h_pos = X[1:-1, 1]
    v_min, v_max = X[1:-1, 1], X[:-2, 1]
    h_min, h_max = X[1:-1, 0], X[2:, 0]
    ax.vlines(v_pos, v_min, v_max, colors="g", linestyles="dashed")
    ax.hlines(h_pos, h_min, h_max, colors="g", linestyles="dashed")


def plot_HVI_cdf(acquisition, value):
    mu, sigma = acquisition.val["F"][0, :], acquisition.val["S"][0, :]
    hvi = HypervolumeImprovement(acquisition.pf, acquisition.rf, mu, sigma)
    x = 10 ** np.linspace(-3, np.log10(hvi.max_hvi), 200)
    probs = [hvi.cdf(v) for v in x]
    # probs2 = [hvi.cdf_monte_carlo(v) for v in x]
    # density = [hvi.pdf(v) * 10 for v in x]

    print(hvi.dominating_prob)
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.semilogx(x, probs, "r--")
    # ax.semilogx(x, density, "g--")
    if 11 < 2:
        epsilon = acquisition.delta_hvi(1 * 0.05)
        ax.vlines(epsilon, np.min(probs), 1 - value, colors="k", linestyles="dashed")
        ax.hlines(1 - value, np.min(x), epsilon, colors="k", linestyles="dashed")
    plt.show()


def visualize_acquisition_landscape(
    surrogate_problem, real_problem, approximation, pareto_front, ref, n_sample=50
):
    # xl = problem.xl.tolist()
    # xu = problem.xu.tolist()
    x = np.linspace(0, 1, n_sample)
    y = np.linspace(0, 1, n_sample)
    x, y = np.meshgrid(x, y)
    X = np.c_[x.ravel(), y.ravel()]
    Y = real_problem.evaluate(X)
    res = surrogate_problem.evaluate(X, return_values_of=["F"], return_as_dictionary=True)
    F = -1.0 * res["F"].reshape(len(x), -1)
    F = F.astype(float)

    Y1 = Y[:, 0].reshape(len(x), -1)
    Y2 = Y[:, 1].reshape(len(x), -1)
    approximation_, idx = find_pareto_front(approximation[:-1, :], return_index=True)
    idx = list(set(range(len(approximation))) - set(idx))

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    plot_attainment_boundary(approximation_, ref, ax)
    ax.plot(pareto_front[:, 0], pareto_front[:, 1], "r--")
    ax.plot(approximation[idx, 0], approximation[idx, 1], "g+")
    ax.plot(approximation_[:, 0], approximation_[:, 1], "g.")
    ax.plot(approximation[-1, 0], approximation[-1, 1], "g*")

    cs = ax.contourf(Y1, Y2, F, cmap="Greys")
    plt.colorbar(cs, ax=ax)
    plt.xlim([0, np.max(approximation[:, 0]) * 1.1])
    plt.ylim([0, np.max(approximation[:, 1]) * 1.1])
    plt.tight_layout()
    plt.show()
    breakpoint()


def main():
    # load arguments
    args, framework_args = get_args()

    # set seed
    np.random.seed(seed)

    # build problem, get initial samples
    problem, pareto_front, X_init, Y_init = build_problem(
        "zdt1", n_decision_vars, n_objectives, n_init_samples, 1
    )
    args.n_var, args.n_obj = problem.n_var, problem.n_obj
    framework_args["solver"]["n_gen"] = 1  # by-pass the optimization

    # initialize optimizer
    optimizer = get_algorithm("pohvi")(
        problem, args.n_iter, args.ref_point, framework_args, random_state=seed
    )

    # save arguments & setup logger
    save_args(args, framework_args)
    # logger = setup_logger(args)
    print(problem, optimizer, sep="\n")

    # optimization
    solution = optimizer.solve(X_init, Y_init)

    for _ in range(args.n_iter):
        # get new design samples and corresponding performance
        X_next, Y_next = next(solution)
        x = np.array([[0.69387755, 0.04081633]])
        # y = optimizer.real_problem.evaluate(x)
        # out = optimizer.current_surrogate.evaluate(x)
        # plot_HVI_cdf(optimizer.acquisition, -out[0][0])
        # print(out)
        visualize_acquisition_landscape(
            optimizer.current_surrogate,
            optimizer.real_problem,
            optimizer.Y,
            pareto_front,
            optimizer.ref_point,
        )


if __name__ == "__main__":
    main()
