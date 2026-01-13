from sympy import Lambda, Symbol, Expr
import sympy2jax

def solve_ivp(
    fun: tuple[list[Symbol], list[Expr], Lambda],
    t_span,
    y0,
    method='RK45',
    t_eval=None,
    dense_output=False,
    events=None,
    vectorized=False,
    args=None,
    **options
):
    pass