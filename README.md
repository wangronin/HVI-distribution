# Hypervolume Improvement Distribution Functions

This is code base of the paper "Probability Distribution of Hypervolume Improvement in Bi-objective Bayesian Optimization",
sumbitted to NeurIPS 2023.

The repository contains two major libraries:

* `hvi/`: the source code to compute the CDF/PDF of HVI. The user should use the class `HypervolumeImprovement`.
* `mobo`: the multi-objective Bayesian optimization algorithm, adapted from project [Diversity-Guided Efficient Multi-Objective Optimization (DGEMO)](https://github.com/yunshengtian/DGEMO)

## Code Structure

```sh
hvi/
 ├── acquisition.py --- acquisition functions
 ├── hv_improvement.py --- probability distribution of HVI
 ├── hypervolume.py --- compute hypervolume 
 └── special math functions used in hv_improvement.py
mobo/
 ├── solver/ --- multi-objective solvers
 ├── surrogate_model/ --- surrogate models
 ├── algorithms.py --- high-level algorithm specifications
 ├── factory.py --- factory for importing different algorithm components
 ├── mobo.py --- main pipeline of multi-objective bayesian optimziation
 ├── selection.py --- selection methods for new samples
 ├── surrogate_problem.py --- multi-objective surrogate problem
 ├── transformation.py --- normalizations on data
 └── utils.py --- utility functions
data/
 ├── data.csv --- data sets used in the NeurIPS submission
 └── CD_plot.R --- R source code for making the critical difference charts
problems/ --- multi-objective problem definitions
scripts/ --- scripts for batch experiments
main.py --- main execution entry for MOBO algorithms
```

## Requirements

* Python version: tested in Python 3.7.7 and 3.10.11
* Install the environment by `pip`:

  ```sh
  pip install -r requirements.txt
  ```

* If the pip installed pymoo is not compiled (will raise warning when running the code), you can clone the pymoo github repository, compile and install this module as described [here](https://pymoo.org/installation.html#development), to gain extra speed-up compared to uncompiled version.

## Basic usage

Run the main file with python with specified arguments:

```python
python main.py --problem zdt1 --algo pohiv --seed 42 --n-iter 10 --n-var 10 --n-obj 2
```
