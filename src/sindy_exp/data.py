from logging import getLogger
from math import ceil
from pathlib import Path
from typing import Any, Callable, Optional, cast
from warnings import warn

import dysts.flows
import dysts.systems
import numpy as np
import scipy
import sympy as sp

from ._dysts_to_sympy import dynsys_to_sympy
from .odes import SHO, CubicHO, Hopf, Kinematics, LotkaVolterra, VanDerPol
from .plotting import plot_training_data
from .typing import Float1D, Float2D, ProbData

try:
    from ._diffrax_solver import _gen_data_jax
except ImportError:
    raise

INTEGRATOR_KEYWORDS = {"rtol": 1e-12, "method": "LSODA", "atol": 1e-12}
TRIALS_FOLDER = Path(__file__).parent.absolute() / "trials"
MOD_LOG = getLogger(__name__)

ODE_CLASSES = {
    klass.lower(): getattr(dysts.flows, klass)
    for klass in dysts.systems.get_attractor_list()
}
ODE_CLASSES.update(
    {
        "lotkavolterra": LotkaVolterra,
        "sho": SHO,
        "cubicho": CubicHO,
        "hopf": Hopf,
        "vanderpol": VanDerPol,
        "kinematics": Kinematics,
    }
)


def _sympy_expr_to_feat_coeff(sp_expr: list[sp.Expr]) -> list[dict[str, float]]:
    expressions = []

    def kv_term(term: sp.Expr) -> tuple[sp.Expr, float]:
        if not isinstance(term, sp.Mul):
            coeff = 1.0
            feat = term
        else:
            try:
                coeff = float(term.args[0])
                args = term.args[1:]
            except TypeError:
                coeff = 1.0
                args = term.args
            if len(args) == 1:
                feat = args[0]
            else:
                feat = sp.Mul(*args)
        return feat, coeff

    for exp in sp_expr:
        expr_dict = {}
        if not isinstance(exp, sp.Add):
            feat, coeff = kv_term(exp)
            expr_dict[feat] = coeff
        else:
            for term in exp.args:
                feat, coeff = kv_term(term)
                expr_dict[feat] = coeff

        expressions.append(expr_dict)
    return expressions


def gen_data(
    system: str,
    seed: Optional[int] = None,
    n_trajectories: int = 1,
    ic_stdev: float = 3,
    noise_abs: Optional[float] = None,
    noise_rel: Optional[float] = None,
    dt: float = 0.01,
    t_end: float = 10,
    display: bool = False,
    array_namespace: str = "numpy",
) -> dict[str, Any]:
    """Generate random training and test data

    An Experiment step according to the mitosis experiment runner.
    Note that test data has no noise.

    Arguments:
        system: the system to integrate
        seed (int): the random seed for number generation
        n_trajectories (int): number of trajectories of training data
        ic_stdev (float): standard deviation for generating initial conditions
        noise_abs (float): measurement noise standard deviation.
            Defaults to .1 if noise_rel is None.
        noise_rel (float): measurement noise-to-signal power ratio.
            Either noise_abs or noise_rel must be None.  Defaults to
            None.
        dt: time step for sample
        t_end: end time of simulation
        display: Whether to display graphics of generated data.

    Returns:
        dictionary of data and descriptive information
    """
    try:
        dyst_sys = ODE_CLASSES[system.lower()]()
    except KeyError as e:
        raise ValueError(
            f"Unknown system {system}.  Check {__name__}.ODE_CLASSES"
        ) from e
    input_features, sp_expr, sp_lambda = dynsys_to_sympy(dyst_sys)
    coeff_true = _sympy_expr_to_feat_coeff(sp_expr)
    rhsfunc = lambda t, X: dyst_sys.rhs(X, t)  # noqa: E731
    try:
        x0_center = dyst_sys.ic
    except KeyError:
        x0_center = np.zeros((len(input_features)), dtype=np.float64)
    try:
        nonnegative = getattr(dyst_sys, "nonnegative", False)
    except KeyError:
        nonnegative = False
    if noise_abs is not None and noise_rel is not None:
        raise ValueError("Cannot specify both noise_abs and noise_rel")
    elif noise_abs is None and noise_rel is None:
        noise_abs = 0.1

    MOD_LOG.info(f"Generating {n_trajectories} trajectories of f{system}")
    if array_namespace == "numpy":
        input_features = [feat.name for feat in input_features]
        dt, t_train, x_train, x_test, x_dot_test, x_train_true, x_train_true_dot = (
            _gen_data(
                rhsfunc,
                len(input_features),
                seed,
                x0_center=x0_center,
                nonnegative=nonnegative,
                n_trajectories=n_trajectories,
                ic_stdev=ic_stdev,
                noise_abs=noise_abs,
                noise_rel=noise_rel,
                dt=dt,
                t_end=t_end,
            )
        )
        integrator = None
    elif array_namespace == "jax":
        try:
            globals()["_gen_data_jax"]
        except KeyError:
            raise ImportError(
                "jax data generation requested but diffrax or sympy2jax not"
                " installed"
            )
        (
            dt,
            t_train,
            x_train,
            x_test,
            x_dot_test,
            x_train_true,
            x_train_true_dot,
            integrator,
        ) = _gen_data_jax(
            (input_features, sp_expr),
            len(input_features),
            seed,
            x0_center=x0_center,
            nonnegative=nonnegative,
            n_trajectories=n_trajectories,
            ic_stdev=ic_stdev,
            noise_abs=noise_abs,
            noise_rel=noise_rel,
            dt=dt,
            t_end=t_end,
        )
        input_features = [feat.name for feat in input_features]
    else:
        raise ValueError(
            f"Unknown array_namespace {array_namespace}.  Must be 'numpy' or 'jax'"
        )
    if display:
        figs = plot_training_data(x_train[0], x_train_true[0])
        figs[0].suptitle("Sample Trajectory")
    return {
        "data": ProbData(
            dt,
            t_train,
            x_train,
            x_test,
            x_dot_test,
            x_train_true,
            x_train_true_dot,
            input_features,
            coeff_true,
            integrator,
        ),
        "main": f"{n_trajectories} trajectories of {rhsfunc}",
        "metrics": {"rel_noise": noise_rel, "abs_noise": noise_abs},
    }


def _gen_data(
    rhs_func: Callable,
    n_coord: int,
    seed: Optional[int],
    n_trajectories: int,
    x0_center: Float1D,
    ic_stdev: float,
    noise_abs: Optional[float],
    noise_rel: Optional[float],
    nonnegative: bool,
    dt: float,
    t_end: float,
) -> tuple[
    float,
    Float1D,
    list[Float2D],
    list[Float2D],
    list[Float2D],
    list[Float2D],
    list[Float2D],
]:
    rng = np.random.default_rng(seed)
    t_train = np.arange(0, t_end, dt)
    t_train_span = (t_train[0], t_train[-1])
    if nonnegative:
        shape = ((x0_center + 1) / ic_stdev) ** 2
        scale = ic_stdev**2 / (x0_center + 1)
        x0_train = np.array(
            [rng.gamma(k, theta, n_trajectories) for k, theta in zip(shape, scale)]
        ).T
        x0_test = np.array(
            [
                rng.gamma(k, theta, ceil(n_trajectories / 2))
                for k, theta in zip(shape, scale)
            ]
        ).T
    else:
        x0_train = ic_stdev * rng.standard_normal((n_trajectories, n_coord)) + x0_center
        x0_test = (
            ic_stdev * rng.standard_normal((ceil(n_trajectories / 2), n_coord))
            + x0_center
        )
    x_train = []
    for traj in range(n_trajectories):
        x_train.append(
            scipy.integrate.solve_ivp(
                rhs_func,
                t_train_span,
                x0_train[traj, :],
                t_eval=t_train,
                **INTEGRATOR_KEYWORDS,
            ).y.T
        )

    def _drop_and_warn(arrs):
        maxlen = max(arr.shape[0] for arr in arrs)

        def _alert_short(arr):
            if arr.shape[0] < maxlen:
                warn(message="Dropping simulation due to blow-up")
                return False
            return True

        arrs = list(filter(_alert_short, arrs))
        if len(arrs) == 0:
            raise ValueError(
                "Simulations failed due to blow-up.  System is too stiff for solver's"
                " numerical tolerance"
            )
        return arrs

    x_train = _drop_and_warn(x_train)
    x_train = np.stack(x_train)
    x_test = []
    for traj in range(ceil(n_trajectories / 2)):
        x_test.append(
            scipy.integrate.solve_ivp(
                rhs_func,
                t_train_span,
                x0_test[traj, :],
                t_eval=t_train,
                **INTEGRATOR_KEYWORDS,
            ).y.T
        )
    x_test = _drop_and_warn(x_test)
    x_test = np.array(x_test)
    x_dot_test = np.array([[rhs_func(0, xij) for xij in xi] for xi in x_test])
    x_train_true = np.copy(x_train)
    x_train_true_dot = np.array(
        [[rhs_func(0, xij) for xij in xi] for xi in x_train_true]
    )
    if noise_rel is not None:
        noise_abs = np.sqrt(_signal_avg_power(x_test) * noise_rel)
    x_train = x_train + cast(float, noise_abs) * rng.standard_normal(x_train.shape)
    x_train = list(x_train)
    x_test = list(x_test)
    x_dot_test = list(x_dot_test)
    x_train_true = list(x_train_true)
    x_train_true_dot = list(x_train_true_dot)
    return dt, t_train, x_train, x_test, x_dot_test, x_train_true, x_train_true_dot


def _max_amplitude(signal: np.ndarray, axis: int) -> float:
    return np.abs(scipy.fft.rfft(signal, axis=axis)[1:]).max() / np.sqrt(
        signal.shape[axis]
    )


def _signal_avg_power(signal: np.ndarray) -> float:
    return np.square(signal).mean()
