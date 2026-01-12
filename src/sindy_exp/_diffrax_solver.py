from sympy import Lambda, Symbol

def solve_ivp(
    fun: tuple[list[Symbol], exprs, Lambda],
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
    