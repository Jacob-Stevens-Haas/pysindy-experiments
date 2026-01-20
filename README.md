# Dynamics Experiments

A library for constructing dynamics experiments.
This includes data generation and plotting/evaluation.


## Getting started

It's not yet on pypi, so install it with `pip install sindy_exp @ git+https://github.com/Jacob-Stevens-Haas/gen-experiments`

Generate data from any of the dysts systems with


## Mitosis

with declarative syntax and
configuration.  These experiments are meant to be run via [`mitosis`](https://github.com/Jacob-Stevens-Haas/mitosis),
but `mitosis` is not a requirement to run them.  It partially annotated with call signatures, and I'm happy to add more
as people point them out.

While this evolved out of the ctf4science and pysindy projects, the aim is to make
these experiments only depend generically on the pysindy API,
without any explicit dependencies.
The experiments can be used for models that are in pysindy but not officially SINDy,
or outside of pysindy so long as they have a similar object.

It's not yet on pypi, so install it with `pip install git+https://github.com/Jacob-Stevens-Haas/gen-experiments`
or clone and install it locally.

## Experiment Steps

There are two experiment steps made available:

* `sindy_exp.data.gen_data()`: calls upon dysts to generate ODE data in standard format
* `sindy_exp.odes.run()`: fits and evaluates a model on data.

If you have data of your own, create a step that puts data in the same format as
`sindy_exp.data.gen_data()`.


## Plotting and diagnostics

Perhaps of most significance are the SINDy diagnostic plotting, which can be used for
your own experiments or in your notebooks.  `sindy_exp.plotting` has a variety
of functions for creating diagnostics of fitted SINDy models:

* `sindy_exp.plotting.compare_coefficient_plots()` and its cousin,
  `sindy_exp.utils.unionize_coeff_matrices()`, which is used to align coefficient matrices from
  models with different features.
* `sindy_exp.utils.coeff_metrics()` and `sindy_exp.utils.pred_metrics()`
* `sindy_exp.plotting.plot_training_data()`
* `sindy_exp.plotting.plot_test_trajectories()`
