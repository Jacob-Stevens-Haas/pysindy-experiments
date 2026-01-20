from math import ceil
from typing import Optional

import diffrax
import jax
import jax.numpy as jnp
import sympy2jax
from sympy import Expr, Symbol

jax.config.update("jax_enable_x64", True)


def _gen_data_jax(
    fun: tuple[list[Symbol], list[Expr]],
    n_coord: int,
    seed: int,
    n_trajectories: int,
    x0_center: jax.Array,
    ic_stdev: float,
    noise_abs: Optional[float],
    noise_rel: Optional[float],
    nonnegative: bool,
    dt: float,
    t_end: float,
) -> tuple[
    float,
    jax.Array,
    list[jax.Array],
    list[jax.Array],
    list[jax.Array],
    list[jax.Array],
    list[jax.Array],
    list[diffrax.Solution],
]:
    symbols = fun[0]
    exprs = fun[1]
    rhstree = sympy2jax.SymbolicModule(exprs)

    def ode_sys(t, state, args):
        return jnp.asarray(
            rhstree(**{str(x_sym): state_i for x_sym, state_i in zip(symbols, state)})
        )

    term = diffrax.ODETerm(ode_sys)
    solver = diffrax.Tsit5()
    save_at = diffrax.SaveAt(ts=jnp.arange(0, t_end, dt), dense=True)

    # Random initialization
    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)
    t_train = jnp.arange(0, t_end, dt)
    if nonnegative:
        shape = ((x0_center + 1) / ic_stdev) ** 2
        scale = ic_stdev**2 / (x0_center + 1)
        x0_train = jnp.array(
            [
                jax.random.gamma(subkey, k, shape=(n_trajectories,)) * theta
                for k, theta in zip(shape, scale)
            ]
        ).T
        key, subkey = jax.random.split(key)
        x0_test = jnp.array(
            [
                jax.random.gamma(subkey, k, shape=(ceil(n_trajectories / 2),)) * theta
                for k, theta in zip(shape, scale)
            ]
        ).T
    else:
        x0_train = (
            ic_stdev * jax.random.normal(subkey, (n_trajectories, n_coord)) + x0_center
        )
        key, subkey = jax.random.split(key)
        x0_test = (
            ic_stdev * jax.random.normal(subkey, (ceil(n_trajectories / 2), n_coord))
            + x0_center
        )
    x_train_true: list[jax.Array] = []
    x_test: list[jax.Array] = []

    # IVPs
    solvers: list[diffrax.Solution] = []
    for traj in range(n_trajectories):
        y0_jax = x0_train[traj]
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=0,
            t1=t_end,
            dt0=dt,  # Initial step size
            y0=y0_jax,
            args=(),
            saveat=save_at,
            max_steps=int(10 * (t_end - 0) / dt),
        )
        x_train_true.append(sol.ys)
        solvers.append(sol)

    for traj in range(ceil(n_trajectories / 2)):
        y0_jax = x0_test[traj]
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=0,
            t1=t_end,
            dt0=dt,  # Initial step size
            y0=y0_jax,
            args=(),
            saveat=save_at,
            max_steps=int(10 * (t_end - 0) / dt),
        )
        x_test.append(sol.ys)
        solvers.append(sol)

    # Measurement noise
    if noise_abs is None:
        assert noise_rel is not None  # force type narrowing
        noise_abs = float(
            jnp.sqrt(_signal_avg_power(jnp.asarray(x_train_true))) * noise_rel
        )

    x_train = [
        x_i + jax.random.normal(key, x_i.shape) * noise_abs for x_i in x_train_true
    ]

    # True Derivatives
    x_dot_test = [ode_sys(0, x_i, None) for x_i in x_test]
    x_train_true_dot = [ode_sys(0, x_i, None) for x_i in x_train_true]

    return (
        dt,
        t_train,
        x_train,
        x_test,
        x_dot_test,
        x_train_true,
        x_train_true_dot,
        solvers,
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
