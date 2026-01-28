from logging import getLogger
from typing import Any, Callable, Optional, cast

import dysts.flows
import dysts.systems
import numpy as np
import scipy

from ._dysts_to_sympy import dynsys_to_sympy
from ._odes import SHO, CubicHO, Hopf, Kinematics, LotkaVolterra, VanDerPol
from ._plotting import plot_training_data
from ._typing import Float1D, ProbData
from ._utils import _sympy_expr_to_feat_coeff

try:
    import jax

    from ._diffrax_solver import _gen_data_jax
except ImportError:
    raise

INTEGRATOR_KEYWORDS = {"rtol": 1e-12, "method": "LSODA", "atol": 1e-12}
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


def gen_data(
    system: str,
    seed: int,
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
    prob_data_list: list[ProbData] = []
    if array_namespace == "numpy":
        feature_names = [feat.name for feat in input_features]
        for _ in range(n_trajectories):
            seed += 1
            prob = _gen_data(
                rhsfunc,
                feature_names,
                seed,
                x0_center=x0_center,
                nonnegative=nonnegative,
                ic_stdev=ic_stdev,
                noise_abs=noise_abs,
                noise_rel=noise_rel,
                dt=dt,
                t_end=t_end,
            )
            prob_data_list.append(prob)
    elif array_namespace == "jax":
        try:
            globals()["_gen_data_jax"]
        except KeyError:
            raise ImportError(
                "jax data generation requested but diffrax or sympy2jax not"
                " installed"
            )
        this_seed = jax.random.PRNGKey(seed)
        for _ in range(n_trajectories):
            this_seed, _ = jax.random.split(this_seed)
            prob = _gen_data_jax(
                sp_expr,
                input_features,
                this_seed,
                x0_center=x0_center,
                nonnegative=nonnegative,
                ic_stdev=ic_stdev,
                noise_abs=noise_abs,
                noise_rel=noise_rel,
                dt=dt,
                t_end=t_end,
            )
            prob_data_list.append(prob)
    else:
        raise ValueError(
            f"Unknown array_namespace {array_namespace}.  Must be 'numpy' or 'jax'"
        )
    if display and prob_data_list:
        sample = prob_data_list[0]
        figs = plot_training_data(sample.t_train, sample.x_train, sample.x_train_true)
        figs[0].suptitle("Sample Trajectory")
    return {
        "data": {"trajectories": prob_data_list, "coeff_true": coeff_true},
        "main": f"{n_trajectories} trajectories of {rhsfunc}",
        "metrics": {"rel_noise": noise_rel, "abs_noise": noise_abs},
    }


def _gen_data(
    rhs_func: Callable,
    input_features: list[str],
    seed: Optional[int],
    x0_center: Float1D,
    ic_stdev: float,
    noise_abs: Optional[float],
    noise_rel: Optional[float],
    nonnegative: bool,
    dt: float,
    t_end: float,
) -> ProbData:
    rng = np.random.default_rng(seed)
    t_train = np.arange(0, t_end, dt)
    t_train_span = (t_train[0], t_train[-1])
    if nonnegative:
        shape = ((x0_center + 1) / ic_stdev) ** 2
        scale = ic_stdev**2 / (x0_center + 1)
        x0 = np.array([rng.gamma(k, theta) for k, theta in zip(shape, scale)])
    else:
        x0 = ic_stdev * rng.standard_normal(len(input_features)) + x0_center
    x_train = scipy.integrate.solve_ivp(
        rhs_func,
        t_train_span,
        x0,
        t_eval=t_train,
        **INTEGRATOR_KEYWORDS,
    ).y.T

    x_train_true = np.copy(x_train)
    x_train_true_dot = np.array([rhs_func(0, xi) for xi in x_train_true])
    if noise_rel is not None:
        noise_abs = np.sqrt(_signal_avg_power(x_train) * noise_rel)
    x_train = x_train + cast(float, noise_abs) * rng.standard_normal(x_train.shape)

    return ProbData(
        dt, t_train, x_train, x_train_true, x_train_true_dot, input_features
    )


def _max_amplitude(signal: np.ndarray, axis: int) -> float:
    return np.abs(scipy.fft.rfft(signal, axis=axis)[1:]).max() / np.sqrt(
        signal.shape[axis]
    )


def _signal_avg_power(signal: np.ndarray) -> float:
    return np.square(signal).mean()
