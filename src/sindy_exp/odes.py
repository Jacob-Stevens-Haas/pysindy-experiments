from importlib import resources
from logging import getLogger
from typing import Callable, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import pysindy as ps
import sympy as sp
from dysts.base import DynSys

from .plotting import (
    compare_coefficient_plots_from_dicts,
    plot_test_trajectory,
    plot_training_data,
)
from .typing import ProbData, _BaseSINDy
from .utils import (
    DynamicsTrialData,
    FullDynamicsTrialData,
    SINDyTrialUpdate,
    coeff_metrics,
    integration_metrics,
    simulate_test_data,
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
LOCAL_DYNAMICS_PATH = resources.files("sindy_exp").joinpath("addl_attractors.json")


def _add_forcing(
    forcing_func: Callable[[float], np.ndarray[tuple[T], DType]],
    auto_func: Callable[
        [float, np.ndarray[tuple[T], DType]], np.ndarray[tuple[T], DType]
    ],
) -> Callable[[float, np.ndarray], np.ndarray]:
    """Add a time-dependent forcing term to a rhs func

    Args:
        forcing_func: The forcing function to add
        auto_func: An existing rhs func for solve_ivp

    Returns:
        A rhs function for integration
    """

    def sum_of_terms(
        t: float, state: np.ndarray[tuple[T], DType]
    ) -> np.ndarray[tuple[T], DType]:
        return np.array(forcing_func(t)) + np.array(auto_func(t, state))

    return sum_of_terms


class LotkaVolterra(DynSys):
    """Lotka-Volterra (predator-prey) dynamical system."""

    nonnegative = True

    def __init__(self):
        super().__init__(metadata_path=LOCAL_DYNAMICS_PATH)

    @staticmethod
    def _rhs(x, y, t: float, alpha, beta, gamma, delta) -> np.ndarray:
        """LV dynamics

        Args:
            x: prey population
            y: predator population
            t: time (ignored, since autonomous)
            alpha: prey growth rate
            beta: predation rate
            delta: predator reproduction rate
            gamma: predator death rate
        """
        dxdt = alpha * x - beta * x * y
        dydt = delta * x * y - gamma * y
        return np.array([dxdt, dydt])


class Hopf(DynSys):
    """Hopf normal form dynamical system."""

    def __init__(self):
        super().__init__(metadata_path=LOCAL_DYNAMICS_PATH)

    @staticmethod
    def _rhs(x, y, t: float, mu, omega, A) -> np.ndarray:
        dxdt = mu * x - omega * y - A * (x**3 + x * y**2)
        dydt = omega * x + mu * y - A * (x**2 * y + y**3)
        return np.array([dxdt, dydt])


class SHO(DynSys):
    """Linear damped simple harmonic oscillator"""

    def __init__(self):
        super().__init__(metadata_path=LOCAL_DYNAMICS_PATH)

    @staticmethod
    def _rhs(x, y, t: float, a, b, c, d) -> np.ndarray:
        dxdt = a * x + b * y
        dydt = c * x + d * y
        return np.array([dxdt, dydt])


class CubicHO(DynSys):
    """Cubic damped harmonic oscillator."""

    def __init__(self):
        super().__init__(metadata_path=LOCAL_DYNAMICS_PATH)

    @staticmethod
    def _rhs(x, y, t: float, a, b, c, d) -> np.ndarray:
        dxdt = a * x**3 + b * y**3
        dydt = c * x**3 + d * y**3
        return np.array([dxdt, dydt])


class VanDerPol(DynSys):
    """Van der Pol oscillator.

    dx/dt = y
    dy/dt = mu * (1 - x^2) * y - x
    """

    def __init__(self):
        super().__init__(metadata_path=LOCAL_DYNAMICS_PATH)

    @staticmethod
    def _rhs(x, x_dot, t: float, mu) -> np.ndarray:
        dxdt = x_dot
        dx2dt2 = mu * (1 - x**2) * x_dot - x
        return np.array([dxdt, dx2dt2])


class Kinematics(DynSys):
    """One-dimensional kinematics with constant acceleration.

    dx/dt = v
    dv/dt = a
    """

    def __init__(self):
        super().__init__(metadata_path=LOCAL_DYNAMICS_PATH)

    @staticmethod
    def _rhs(x, v, t: float, a) -> np.ndarray:
        dxdt = v
        dvdt = a
        return np.array([dxdt, dvdt])


def fit_eval(
    data: tuple[list[ProbData], list[dict[sp.Expr, float]]],
    model: _BaseSINDy,
    simulations: bool = True,
    display: bool = True,
    return_all: bool = True,
) -> dict | tuple[dict, DynamicsTrialData | FullDynamicsTrialData]:
    """Fit and evaluate a SINDy model on a set of trajectories.

    Args:
        data: Tuple of (trajectories, true_equations), where ``trajectories`` is
            a list of ProbData objects and ``true_equations`` is a list of
            dictionaries mapping SymPy symbols to their true coefficients for
            each state coordinate.
        model: A SINDy-like model implementing the _BaseSINDy protocol.
        simulations: Whether to run forward simulations for evaluation.
        display: Whether to generate plots as part of evaluation.
        return_all: If True, return a dictionary containing metrics and the
            assembled DynamicsTrialData; otherwise return only the metrics
            dictionary.
    """

    trajectories, true_equations = data
    input_features = trajectories[0].input_features

    x_train = [traj.x_train for traj in trajectories]
    t_train = [traj.t_train for traj in trajectories]
    model.fit(x_train, t=t_train, feature_names=input_features)

    MOD_LOG.info(f"Fitting a model: {model}")
    coeff_true_dicts, coeff_est_dicts = unionize_coeff_dicts(model, true_equations)
    if isinstance(model.feature_library, ps.WeakPDELibrary):
        # WeakPDE library fails to simulate, so insert nonweak library
        # to Pipeline and SINDy model.
        inner_lib = model.feature_library.function_library
        model.feature_library = inner_lib  # type: ignore  # TODO: Fix in pysindy
    if isinstance(model, ps.SINDy) and hasattr(
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
            sim = simulate_test_data(model, traj.dt, traj.x_train_true)
            sims.append(sim)
            integration_metric_list.append(
                integration_metrics(
                    model,
                    traj.x_train_true,
                    traj.t_train,
                    traj.x_train_true_dot,
                )
            )

        agg_integration_metrics: dict[str, float | np.floating] = {}
        for key in integration_metric_list[0].keys():
            values = [m[key] for m in integration_metric_list]
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
    if return_all:
        return {"metrics": metrics, "data": trial_data, "main": metrics["main"]}
    return metrics


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
