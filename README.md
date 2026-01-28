# Dynamics Experiments

A library for constructing dynamics experiments.
This includes data generation and plotting/evaluation.

## Getting started

It's not yet on PyPI, so install it with `pip install sindy_exp @ git+https://github.com/Jacob-Stevens-Haas/sindy-experiments`

Generate data

    data = sindy_exp.data.gen_data("lorenz", num_trajectories=5, t_end=10.0, dt=0.01)["data]

Evaluate your SINDy-like model with:

    sindy_exp.odes.fit_eval(model, data)

![Coefficient plots](images/coeff.png)

A list of available ODE systems can be found in `ODE_CLASSES`, which includes most
of the systems from the [dysts package](https://pypi.org/project/dysts/) as well as some non-chaotic systems.

## ODE representation

We deal primarily with autonomous ODE systems of the form:

    dx/dt = sum_i f_i(x)

Thus, we represent ODE systems as a list of right-hand side expressions.
Each element is a dictionary mapping a term (Sympy expression) to its coefficient.

## Other useful imports, compatibility, and extensions

This is built to be compatible with dynamics learning models that follow the
pysindy _BaseSINDy interface.
The experiments are also built to be compatible with the `mitosis` tool,
an experiment runner.
To integrate your own experiments or data generation in a way that is compatible,
see the `ProbData` and `DynamicsTrialData` classes.
For plotting tools, see `plot_coefficients`, `compare_coefficient_plots_from_dicts`,
`plot_test_trajectory`, `plot_training_data`, and `COLOR`.
For metrics, see `coeff_metrics`, `pred_metrics`, and `integration_metrics`.

![3d plot](images/composite.png)
![1d plot](images/1d.png)
