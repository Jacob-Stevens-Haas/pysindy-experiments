from importlib import resources
from logging import getLogger
from typing import Callable, Optional, TypeVar, cast
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pysindy as ps
from dysts.base import DynSys

from .plotting import (
    compare_coefficient_plots,
    plot_test_trajectories,
    plot_training_data,
)
from .typing import ProbData, _BaseSINDy
from .utils import (
    DynamicsTrialData,
    FullSINDyTrialData,
    coeff_metrics,
    integration_metrics,
    make_model,
    simulate_test_data,
    unionize_coeff_matrices,
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


def add_forcing(
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


def run(
    data: ProbData,
    diff_params: Optional[dict] = None,
    feat_params: Optional[dict] = None,
    opt_params: Optional[dict] = None,
    model: Optional[_BaseSINDy] = None,
    display: bool = True,
    return_all: bool = False,
) -> dict | tuple[dict, DynamicsTrialData | FullSINDyTrialData]:
    input_features = data.input_features
    dt = data.dt
    x_train = data.x_train
    t_train = data.t_train
    x_train_true = data.x_train_true
    x_test = data.x_test
    x_dot_test = data.x_dot_test
    coeff_true = data.coeff_true

    if isinstance(feat_params, dict) and feat_params["featcls"] == "weak":
        feat_params.pop("featcls")
        feat_params = ps.WeakPDELibrary(**feat_params, spatiotemporal_grid=data.t_train)

    if model is None and all(
        (diff_params is not None, feat_params is not None, opt_params is not None)
    ):
        warn(
            "Passing the built model directly is now accepted.  This is the recommended"
            "way, and passing (nested) dictionaries to build classes is likely to be"
            "deprecated.",
            PendingDeprecationWarning,
        )
        model = make_model(input_features, dt, diff_params, feat_params, opt_params)
    elif (
        model is not None
        and any(
            (diff_params is not None, feat_params is not None, opt_params is not None)
        )
        or model is None
        and any((diff_params is None, feat_params is None, opt_params is None))
    ):
        raise ValueError(
            "Either model must be None and the builder dictionaries all defined,"
            "or vice versa."
        )
    model = cast(_BaseSINDy, model)
    model.feature_names = data.input_features
    model.fit(x_train, t=len(x_train) * [t_train])
    MOD_LOG.info(f"Fitting a model: {model}")
    coeff_true, coefficients, feature_names = unionize_coeff_matrices(
        model, (data.input_features, coeff_true)
    )
    if isinstance(model.feature_library, ps.WeakPDELibrary):
        # WeakPDE library fails to simulate, so insert nonweak library
        # to Pipeline and SINDy model.
        inner_lib = model.feature_library.function_library
        model.model.steps[0] = ("features", inner_lib)
        model.feature_library = inner_lib  # type: ignore  # TODO: Fix in pysindy
    sim_ind = -1
    if isinstance(model, ps.SINDy) and hasattr(
        model.differentiation_method, "smoothed_x_"
    ):
        smooth_x = model.differentiation_method.smoothed_x_
    else:  # using WeakPDELibrary
        smooth_x = x_train[0]
    trial_data: DynamicsTrialData = {
        "dt": dt,
        "coeff_true": coeff_true,
        "coeff_fit": coefficients,
        "feature_names": feature_names,
        "input_features": input_features,
        "t_train": t_train,
        "x_true": x_train_true[sim_ind],
        "x_train": x_train[sim_ind],
        "smooth_train": smooth_x,
        "x_test": x_test[sim_ind],
        "x_dot_test": x_dot_test[sim_ind],
        "model": model,
    }
    if display:
        MOD_LOG.info(f"Simulating a model: {model}")
        trial_data: FullSINDyTrialData = trial_data | simulate_test_data(
            trial_data["model"], trial_data["dt"], trial_data["x_test"]
        )
        plot_ode_panel(trial_data)
    MOD_LOG.info(f"Evaluating a model: {model}")
    metrics = coeff_metrics(coefficients, coeff_true)
    metrics.update(integration_metrics(model, x_test, t_train, x_dot_test))
    if return_all:
        return {"metrics": metrics, "data": trial_data, "main": metrics["main"]}
    return metrics


def ablate_feat(
    data: ProbData,
    diff_params: dict,
    feat_params: dict,
    opt_params: dict,
    display: bool = True,
    return_all: bool = False,
) -> dict | tuple[dict, DynamicsTrialData | FullSINDyTrialData]:
    """Like run(), but hide one input feature from model

    Temporary and highly WET.
    """
    input_features = data.input_features[:-1]
    dt = data.dt
    x_train = [x[..., :-1] for x in data.x_train]
    t_train = data.t_train
    x_train_true = [x[..., :-1] for x in data.x_train_true]
    x_test = [x[..., :-1] for x in data.x_test]
    x_dot_test = [x[..., :-1] for x in data.x_dot_test]

    if feat_params["featcls"] == "weak":
        feat_params.pop("featcls")
        feat_params = ps.WeakPDELibrary(**feat_params, spatiotemporal_grid=data.t_train)

    model = make_model(input_features, dt, diff_params, feat_params, opt_params)
    model.fit(x_train, t=dt)
    MOD_LOG.info(f"Fitting a model: {model}")
    coeff_true, coefficients, feature_names = unionize_coeff_matrices(
        model, (data.input_features[:-1], data.coeff_true[:-1])
    )
    if isinstance(model.feature_library, ps.WeakPDELibrary):
        # WeakPDE library fails to simulate, so insert nonweak library
        # to Pipeline and SINDy model.
        inner_lib = model.feature_library.function_library
        model.model.steps[0] = ("features", inner_lib)
        model.feature_library = inner_lib
    sim_ind = -1
    if hasattr(model.differentiation_method, "smoothed_x_"):
        smooth_x = model.differentiation_method.smoothed_x_
    else:  # using WeakPDELibrary
        smooth_x = x_train[0]
    trial_data: DynamicsTrialData = {
        "dt": dt,
        "coeff_true": coeff_true,
        "coeff_fit": coefficients,
        "feature_names": feature_names,
        "input_features": input_features,
        "t_train": t_train,
        "x_true": x_train_true[sim_ind],
        "x_train": x_train[sim_ind],
        "smooth_train": smooth_x,
        "x_test": x_test[sim_ind],
        "x_dot_test": x_dot_test[sim_ind],
        "model": model,
    }
    if display:
        MOD_LOG.info(f"Simulating a model: {model}")
        trial_data: FullSINDyTrialData = trial_data | simulate_test_data(
            trial_data["model"], trial_data["dt"], trial_data["x_test"]
        )
        plot_ode_panel(trial_data)
    MOD_LOG.info(f"Evaluating a model: {model}")
    metrics = coeff_metrics(coefficients, coeff_true)
    metrics.update(integration_metrics(model, x_test, t_train, x_dot_test))
    if return_all:
        return {"metrics": metrics, "data": trial_data, "main": metrics["main"]}
    return metrics


def plot_ode_panel(trial_data: FullSINDyTrialData):
    trial_data["model"].print()
    plot_training_data(
        trial_data["x_train"], trial_data["x_true"], trial_data["smooth_train"]
    )
    compare_coefficient_plots(
        trial_data["coeff_fit"],
        trial_data["coeff_true"],
        input_features=[_texify(feat) for feat in trial_data["input_features"]],
        feature_names=[_texify(feat) for feat in trial_data["feature_names"]],
    )
    plot_test_trajectories(
        trial_data["x_test"],
        trial_data["x_sim"],
        trial_data["t_test"],
        trial_data["t_sim"],
    )
    plt.show()


def _texify(input: str) -> str:
    if input[0] != "$":
        input = "$" + input
    if input[-1] != "$":
        input = input + "$"
    return input
