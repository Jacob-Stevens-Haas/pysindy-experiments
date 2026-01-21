from typing import Optional

import diffrax
import jax
import jax.numpy as jnp
import sympy2jax
from sympy import Expr, Symbol

from .typing import ProbData

jax.config.update("jax_enable_x64", True)


def _gen_data_jax(
    exprs: list[Expr],
    input_features: list[Symbol],
    seed: jax.Array,
    x0_center: jax.Array,
    ic_stdev: float,
    noise_abs: Optional[float],
    noise_rel: Optional[float],
    nonnegative: bool,
    dt: float,
    t_end: float,
) -> ProbData:
    rhstree = sympy2jax.SymbolicModule(exprs)

    def ode_sys(t, state, args):
        return jnp.asarray(
            rhstree(
                **{
                    str(x_sym): state_i
                    for x_sym, state_i in zip(input_features, state, strict=True)
                }
            )
        )

    term = diffrax.ODETerm(ode_sys)
    solver = diffrax.Tsit5()
    save_at = diffrax.SaveAt(ts=jnp.arange(0, t_end, dt), dense=True)

    # Random initialization
    key, subkey = jax.random.split(seed)
    t_train = jnp.arange(0, t_end, dt)
    if nonnegative:
        shape = ((x0_center + 1) / ic_stdev) ** 2
        scale = ic_stdev**2 / (x0_center + 1)
        x0 = jnp.array(
            jax.random.gamma(subkey, k) * theta for k, theta in zip(shape, scale)
        ).T

    else:
        x0 = ic_stdev * jax.random.normal(subkey, (len(input_features),)) + x0_center
    key, subkey = jax.random.split(key)

    # IVPs
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0,
        t1=t_end,
        dt0=dt,  # Initial step size
        y0=x0,
        args=(),
        saveat=save_at,
        max_steps=int(10 * (t_end - 0) / dt),
    )
    x_train_true: jax.Array = sol.ys  # type: ignore

    # Measurement noise
    if noise_abs is None:
        assert noise_rel is not None  # force type narrowing
        noise_abs = float(jnp.sqrt(_signal_avg_power(x_train_true)) * noise_rel)

    x_train = x_train_true + jax.random.normal(key, x_train_true.shape) * noise_abs

    # True Derivatives
    x_train_true_dot = jnp.array([ode_sys(0, xi, None) for xi in x_train_true])

    stringy_features = [sym.name for sym in input_features]
    return ProbData(
        dt, t_train, x_train, x_train_true, x_train_true_dot, stringy_features, sol
    )


def _signal_avg_power(signal: jax.Array) -> jax.Array:
    return jnp.square(signal).mean()


## %  # noqa:E266
if __name__ == "__main__":
    # Debug example
    from sindy_exp.data import gen_data

    data_dict = gen_data(
        "valliselnino",
        seed=50,
        n_trajectories=1,
        ic_stdev=3,
        noise_rel=0.1,
        display=True,
        array_namespace="jax",
    )
    print(data_dict["input_features"])
