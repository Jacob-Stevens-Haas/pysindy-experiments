from typing import Optional

from sympy import Lambda, Symbol, Expr
import sympy2jax

from .typing import Float1D, Float2D

def _gen_data_jax(
    fun: tuple[list[Symbol], list[Expr], Lambda],
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
    pass