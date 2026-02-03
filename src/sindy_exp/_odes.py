from logging import getLogger
from typing import Any, Literal, TypeVar, cast, overload

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from ._plotting import (
    compare_coefficient_plots_from_dicts,
    plot_test_trajectory,
    plot_training_data,
)
from ._typing import (
    DynamicsTrialData,
    ExperimentResult,
    FullDynamicsTrialData,
    SimProbData,
    SINDyTrialUpdate,
    _BaseSINDy,
)
from ._utils import (
    _simulate_test_data,
    coeff_metrics,
    integration_metrics,
    unionize_coeff_dicts,
)

metric_ordering = {
    "coeff_precision": "max",
    "coeff_f1": "max",
    "coeff_recall": "max",
    "coeff_mae": "min",
    "coeff_mse": "min",
    "mse_plot": "min",
    "mae_plot": "min",
}


T = TypeVar("T", bound=int)
DType = TypeVar("DType", bound=np.dtype)
MOD_LOG = getLogger(__name__)


@overload
def fit_eval(
    data: tuple[list[SimProbData], list[dict[sp.Expr, float]]],
    model: _BaseSINDy,
    simulations: Literal[False],
    display: bool,
) -> ExperimentResult[DynamicsTrialData]: ...


@overload
def fit_eval(
    data: tuple[list[SimProbData], list[dict[sp.Expr, float]]],
    model: _BaseSINDy,
    simulations: Literal[True],
    display: bool,
) -> ExperimentResult[FullDynamicsTrialData]: ...


def fit_eval(
    data: tuple[list[SimProbData], list[dict[sp.Expr, float]]],
    model: Any,
    simulations: bool = True,
    display: bool = True,
) -> ExperimentResult:
    """Fit and evaluate a SINDy model on a set of trajectories.

    Args:
        data: Tuple of (trajectories, true_equations), where ``trajectories`` is
            a list of SimProbData objects and ``true_equations`` is a list of
            dictionaries mapping SymPy symbols to their true coefficients for
            each state coordinate.
        model: A SINDy-like model implementing the _BaseSINDy protocol.
        simulations: Whether to run forward simulations for evaluation.
        display: Whether to generate plots as part of evaluation.
    """
    model = cast(_BaseSINDy, model)
    for trajectory in data[0]:
        assert trajectory.x_train_true is not None
    trajectories, true_equations = data
    input_features = trajectories[0].input_features

    x_train = [traj.x_train for traj in trajectories]
    t_train = [traj.t_train for traj in trajectories]
    model.fit(x_train, t=t_train, feature_names=input_features)

    MOD_LOG.info(f"Fitting a model: {model}")
    coeff_true_dicts, coeff_est_dicts = unionize_coeff_dicts(model, true_equations)

    # Special workaround for pysindy's legacy WeakPDELibrary
    if hasattr(model.feature_library, "K"):
        # WeakPDE library fails to simulate, so insert nonweak library
        # to Pipeline and SINDy model.
        inner_lib = model.feature_library.function_library
        model.feature_library = inner_lib  # type: ignore  # TODO: Fix in pysindy

    # Special workaround for pysindy's bad (soon to be legacy) differentiation API
    if hasattr(model, "differentiation_method") and hasattr(
        model.differentiation_method, "smoothed_x_"
    ):
        smooth_x = []
        for traj in trajectories:
            model.differentiation_method(traj.x_train, t=traj.t_train)
            smooth_x.append(model.differentiation_method.smoothed_x_)
    else:  # using WeakPDELibrary
        smooth_x = x_train

    trial_data = DynamicsTrialData(
        trajectories=trajectories,
        true_equations=coeff_true_dicts,
        sindy_equations=coeff_est_dicts,
        model=model,
        input_features=input_features,
        smooth_train=smooth_x,
    )
    MOD_LOG.info(f"Evaluating a model: {model}")
    metrics = coeff_metrics(coeff_est_dicts, coeff_true_dicts)
    if simulations:
        sims: list[SINDyTrialUpdate] = []
        integration_metric_list: list[dict[str, float | np.floating]] = []
        for traj in trajectories:
            sim = _simulate_test_data(model, traj.t_train, traj.x_train_true)
            sims.append(sim)
            integration_metric_list.append(
                integration_metrics(
                    model,
                    traj.x_train_true,
                    traj.t_train,
                    traj.x_train_true_dot,
                )
            )

        agg_integration_metrics: dict[str, float] = {}
        for key in integration_metric_list[0].keys():
            values = cast(list[float], [m[key] for m in integration_metric_list])
            agg_integration_metrics[key] = float(np.mean(values))
        metrics.update(agg_integration_metrics)

        trial_data = FullDynamicsTrialData(sims=sims, **trial_data.__dict__)
    if display:
        plot_ode_panel(trial_data)
        for i, traj in enumerate(trajectories):
            fig_composite, fig_by_coord_1d = plot_training_data(
                traj.t_train,
                traj.x_train,
                traj.x_train_true,
                x_smooth=smooth_x[i],
                coord_names=input_features,
            )
            if simulations:
                # Overlay test trajectory time series on the coordinate-wise figure
                plot_test_trajectory(
                    traj.x_train_true,
                    sims[i].x_sim,
                    traj.t_train,
                    sims[i].t_sim,
                    figs=(fig_composite, fig_by_coord_1d),
                    coord_names=input_features,
                )

    return {"metrics": metrics, "data": trial_data, "main": metrics["main"]}


def plot_ode_panel(trial_data: DynamicsTrialData):
    trial_data.model.print()
    compare_coefficient_plots_from_dicts(
        trial_data.sindy_equations,
        trial_data.true_equations,
        input_features=[_texify(feat) for feat in trial_data.input_features],
    )
    plt.show()


def _texify(input: str) -> str:
    if input[0] != "$":
        input = "$" + input
    if input[-1] != "$":
        input = input + "$"
    return input
