# %%
import importlib

import gen_experiments.utils as utils

utils = importlib.reload(utils)

# %%
utils.plot_summary_metric(
    "coeff_mae",
    "sim_params.t_end",
    ("cubic_ho", "e3399c"),
    ("duff", "c8d97a"),
    ("hopf", "8aa111"),
    ("lv", "393bde"),
    ("sho", "46e3e1"),
    ("vdp", "580e49"),
    ("ross", "2e7508"),
)
pass
# %%
results = utils.load_results("580e49")
params = results["plot_data"][0]["params"]
# %%
utils.plot_point_across_experiments(
    params,
    ("Cubic Oscillator", "e3399c"),
    ("Duffing", "c8d97a"),
    ("Hopf", "8aa111"),
    ("Lotka-Volterra", "393bde"),
    ("Harmonic Oscillator", "46e3e1"),
    ("Van der Pol", "580e49"),
    ("Rossler", "2e7508"),
    style="test",
)
pass
# %%
utils.plot_experiment_across_gridpoints(
    "2e7508",
    ("Kalman", results["plot_data"][0]["params"]),
    ("TV", results["plot_data"][3]["params"]),
    ("SavGol", results["plot_data"][6]["params"]),
    style="test"
)
utils.plot_experiment_across_gridpoints(
    "580e49",
    ("Kalman", results["plot_data"][0]["params"]),
    ("TV", results["plot_data"][3]["params"]),
    ("SavGol", results["plot_data"][6]["params"]),
    style="training"
)
pass
# %%
utils = importlib.reload(utils)
results = utils.load_results("580e49")
params1 = results["plot_data"][0]["params"]
params2 = results["plot_data"][3]["params"]
params3 = results["plot_data"][6]["params"]
utils.plot_summary_test_train(
    [
        ("Cubic Oscillator", "e3399c"),
        ("Duffing", "c8d97a"),
        ("Hopf", "8aa111"),
        ("Lotka-Volterra", "393bde"),
        ("Harmonic Oscillator", "46e3e1"),
        ("Van der Pol", "580e49"),
        ("Rossler", "2e7508")
    ],
    [("Kalman", params1), ("TV", params2), ("SavGol", params3)],
    style="test"
)
pass
# %%
