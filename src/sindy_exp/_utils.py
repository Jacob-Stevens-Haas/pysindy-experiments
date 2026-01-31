import logging
from itertools import chain
from typing import cast
from warnings import warn

import numpy as np
import pysindy as ps
import sklearn
import sklearn.metrics
import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from sindy_exp._typing import SINDyTrialUpdate

from ._typing import Float1D, Float2D, _BaseSINDy

logger = logging.getLogger(__name__)


def _sympy_expr_to_feat_coeff(sp_expr: list[sp.Expr]) -> list[dict[sp.Expr, float]]:
    """Convert symbolic rhs expressions into feature/coeff dictionaries.

    Each expression is assumed to be a sum of terms; each term is either a
    simple SymPy expression or a product of a numeric coefficient and a
    feature expression. The output is a list of dictionaries, one per
    expression, mapping feature expressions to their numeric coefficients.
    """

    expressions: list[dict[sp.Expr, float]] = []

    def kv_term(term: sp.Expr) -> tuple[sp.Expr, float]:
        if not isinstance(term, sp.Mul):
            # Term is either a constant or a feature without a coefficient
            if isinstance(term, sp.Number):
                coeff = float(term)
                feat = sp.Integer(1)
            elif isinstance(term, (sp.Symbol, sp.Pow, sp.Function)):
                coeff = 1.0
                feat = term
            else:
                raise ValueError(f"Unrecognized term format: {term}")
        else:
            try:
                # Assume multiplication is coefficient * feature(s)
                coeff = float(term.args[0])
                args = term.args[1:]
            except TypeError:
                # e.g. x**2 or x * y has no numeric coefficient
                coeff = 1.0
                args = term.args
            if len(args) == 1:
                feat = args[0]
            else:
                feat = sp.Mul(*args)
        return feat, coeff

    for exp in sp_expr:
        expr_dict: dict[sp.Expr, float] = {}
        if not isinstance(exp, sp.Add):
            feat, coeff = kv_term(exp)
            expr_dict[feat] = coeff
        else:
            for term in exp.args:
                feat, coeff = kv_term(term)
                expr_dict[feat] = coeff

        expressions.append(expr_dict)
    return expressions


def _sindy_equations_to_sympy(model: _BaseSINDy) -> tuple[list[sp.Expr], list[sp.Expr]]:
    """Convert a SINDy model's string equations to SymPy expressions.

    Uses sympy's parser with ``convert_xor`` so that terms like ``x^2`` are
    interpreted as powers rather than bitwise XOR.
    """

    # Use a fixed precision for reproducible string equations.
    input_features = {feat: sp.sympify(feat) for feat in model.feature_names}
    eq_strings = model.equations(10)  # type: ignore[call-arg]
    feat_strs = model.feature_library.get_feature_names(
        input_features=model.feature_names
    )
    xforms = standard_transformations + (
        implicit_multiplication_application,
        convert_xor,
    )
    feat_symb = [
        parse_expr(fstr, transformations=xforms, evaluate=False) for fstr in feat_strs
    ]
    eq_symb = [
        parse_expr(
            eq, local_dict=input_features, transformations=xforms, evaluate=False
        )
        for eq in eq_strings
    ]
    return feat_symb, eq_symb


def diff_lookup(kind):
    normalized_kind = kind.lower().replace(" ", "")
    if normalized_kind == "finitedifference":
        return ps.FiniteDifference
    if normalized_kind == "smoothedfinitedifference":
        return ps.SmoothedFiniteDifference
    elif normalized_kind == "sindy":
        return ps.SINDyDerivative
    else:
        raise ValueError


def feature_lookup(kind):
    normalized_kind = kind.lower().replace(" ", "")
    if normalized_kind is None:
        return ps.PolynomialLibrary
    elif normalized_kind == "polynomial":
        return ps.PolynomialLibrary
    elif normalized_kind == "fourier":
        return ps.FourierLibrary
    elif normalized_kind == "weak":
        return ps.WeakPDELibrary
    elif normalized_kind == "pde":
        return ps.PDELibrary
    else:
        raise ValueError


def opt_lookup(kind):
    normalized_kind = kind.lower().replace(" ", "")
    if normalized_kind == "stlsq":
        return ps.STLSQ
    elif normalized_kind == "sr3":
        return ps.SR3
    elif normalized_kind == "miosr":
        return ps.MIOSR
    elif normalized_kind == "trap":
        return ps.TrappingSR3
    elif normalized_kind == "ensemble":
        return ps.EnsembleOptimizer
    else:
        raise ValueError


def coeff_metrics(
    coeff_est_dicts: list[dict[sp.Expr, float]],
    coeff_true_dicts: list[dict[sp.Expr, float]],
) -> dict[str, float | np.floating]:
    """Compute coefficient metrics from aligned coefficient dictionaries.

    Both arguments are expected to be lists of coefficient dictionaries sharing
    the same SymPy-expression keys, such as the output of ``unionize_coeff_dicts``.
    """

    if not coeff_true_dicts or not coeff_est_dicts:
        raise ValueError("Coefficient dictionaries must be non-empty")
    if len(coeff_true_dicts) != len(coeff_est_dicts):
        raise ValueError("True and estimated coefficients must have same length")

    features = list(coeff_true_dicts[0].keys())
    n_coord = len(coeff_true_dicts)
    n_feat = len(features)

    coeff_true = np.zeros((n_coord, n_feat), dtype=float)
    coefficients = np.zeros_like(coeff_true)

    for row_ind, (true_row, est_row) in enumerate(
        zip(coeff_true_dicts, coeff_est_dicts)
    ):
        if set(true_row.keys()) != set(features) or set(est_row.keys()) != set(
            features
        ):
            raise ValueError(
                "Coefficient dictionaries are not aligned across coordinates"
            )
        for col_ind, feat in enumerate(features):
            coeff_true[row_ind, col_ind] = true_row[feat]
            coefficients[row_ind, col_ind] = est_row[feat]

    metrics: dict[str, float | np.floating] = {}
    metrics["coeff_precision"] = float(
        sklearn.metrics.precision_score(
            coeff_true.flatten() != 0, coefficients.flatten() != 0
        )
    )
    metrics["coeff_recall"] = float(
        sklearn.metrics.recall_score(
            coeff_true.flatten() != 0, coefficients.flatten() != 0
        )
    )
    metrics["coeff_f1"] = float(
        sklearn.metrics.f1_score(coeff_true.flatten() != 0, coefficients.flatten() != 0)
    )
    metrics["coeff_mse"] = sklearn.metrics.mean_squared_error(
        coeff_true.flatten(), coefficients.flatten()
    )
    metrics["coeff_mae"] = sklearn.metrics.mean_absolute_error(
        coeff_true.flatten(), coefficients.flatten()
    )
    metrics["main"] = metrics["coeff_f1"]
    return metrics


def pred_metrics(
    model: _BaseSINDy, x_test: np.ndarray, x_dot_test: np.ndarray
) -> dict[str, np.ndarray | float | np.floating]:
    preds = model.predict(x_test)
    err = preds - x_dot_test
    return {
        "pred_l2_fro": (np.linalg.norm(err) / np.linalg.norm(x_dot_test)),
        "pred_l2_each": (np.linalg.norm(err) / np.linalg.norm(x_dot_test)),
        "pred_r2": sklearn.metrics.r2_score(x_dot_test, preds),
    }


def integration_metrics(model: _BaseSINDy, x_test, t_train, x_dot_test):
    metrics = {}
    metrics["mse-plot"] = model.score(
        x_test,
        t_train,
        x_dot_test,
        metric=sklearn.metrics.mean_squared_error,
    )
    metrics["mae-plot"] = model.score(
        x_test,
        t_train,
        x_dot_test,
        metric=sklearn.metrics.mean_absolute_error,
    )
    return metrics


def unionize_coeff_dicts(
    model: _BaseSINDy,
    true_equations: list[dict[sp.Expr, float]],
) -> tuple[list[dict[sp.Expr, float]], list[dict[sp.Expr, float]]]:
    """Align true and estimated coefficient dictionaries using SymPy expressions.

    This function compares the symbolic features present in the true
    equations and in the fitted SINDy model's equations and returns two
    lists of coefficient dictionaries with identical feature keys. Missing
    features in either source are filled with a coefficient of 0.0.

    Args:
        model: Fitted SINDy-like model.
        true_equations: List of coefficient dictionaries mapping SymPy
            expressions to true coefficients, one dict per state coordinate.

    Returns:
        A pair ``(true_aligned, est_aligned)`` where each element is a list
        of coefficient dictionaries ``dict[sp.Expr, float]`` with identical
        keys across all coordinates.
    """

    feat_exprs, est_eq_exprs = _sindy_equations_to_sympy(model)
    # est_equations should be unnecessary; all terms are provided by features
    est_equations = _sympy_expr_to_feat_coeff(est_eq_exprs)

    if len(est_equations) != len(true_equations):
        raise ValueError(
            "True equations and estimated equations must have"
            " the same number of coordinates"
        )

    true_aligned: list[dict[sp.Expr, float]] = []
    est_aligned: list[dict[sp.Expr, float]] = []
    empty_feats: list[dict[sp.Expr, float]] = [dict.fromkeys(feat_exprs, 0.0)]

    all_features = [
        expr
        for eq in chain(empty_feats, true_equations, est_equations)
        for expr in eq.keys()
    ]
    all_features = list(dict.fromkeys(all_features))  # deduplicate, preserve order
    for true_eq, est_eq in zip(true_equations, est_equations):
        true_aligned.append({feat: true_eq.get(feat, 0.0) for feat in all_features})
        est_aligned.append({feat: est_eq.get(feat, 0.0) for feat in all_features})

    return true_aligned, est_aligned


def _simulate_test_data(
    model: _BaseSINDy, dt: float, x_test: Float2D
) -> SINDyTrialUpdate:
    """Add simulation data to grid_data

    This includes the t_sim and x_sim keys.  Does not mutate argument.
    Returns:
        Complete GridPointData
    """
    t_test = cast(Float1D, np.arange(0, len(x_test) * dt, step=dt))
    t_sim = t_test
    try:

        def quit(t, x):
            return np.abs(x).max() - 1000

        quit.terminal = True  # type: ignore
        x_sim = cast(
            Float2D,
            model.simulate(
                x_test[0],
                t_test,
                integrator_kws={
                    "method": "LSODA",
                    "rtol": 1e-12,
                    "atol": 1e-12,
                    "events": [quit],
                },
            ),
        )
    except ValueError:
        warn(message="Simulation blew up; returning zeros")
        x_sim = np.zeros_like(x_test)
    # truncate if integration returns wrong number of points
    t_sim = cast(Float1D, t_test[: len(x_sim)])
    return SINDyTrialUpdate(t_sim=t_sim, t_test=t_test, x_sim=x_sim)


def _drop_and_warn(arrs):
    """Drop trajectories that blew up during simulation"""
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
