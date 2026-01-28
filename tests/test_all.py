from typing import Sequence

import jax
import numpy as np
import pysindy as ps
import pytest
import sympy as sp
from sympy.parsing.sympy_parser import T

XFORMS = T[:6]
jax.config.update("jax_platform_name", "cpu")  # diffrax issue 722
# Alsorequires successive E402


from sindy_exp._data import gen_data  # noqa: E402
from sindy_exp._plotting import _coeff_dicts_to_matrix  # noqa: E402
from sindy_exp._typing import NestedDict  # noqa: E402
from sindy_exp._utils import unionize_coeff_dicts  # noqa: E402
from sindy_exp._utils import (  # noqa: E402
    _sindy_equations_to_sympy,
    _sympy_expr_to_feat_coeff,
)


@pytest.fixture
def jax_cpu_only():
    with jax.default_device(jax.devices("cpu")[0]):
        yield


def test_flatten_nested_dict():
    deep = NestedDict(a=NestedDict(b=1))
    result = deep.flatten()
    assert deep != result
    expected = {"a.b": 1}
    assert result == expected


def test_flatten_nested_bad_dict():
    nested = {1: NestedDict(b=1)}
    # Testing the very thing that causes a typing error, thus ignoring
    with pytest.raises(TypeError, match="keywords must be strings"):
        NestedDict(**nested)  # type: ignore
    with pytest.raises(TypeError, match="Only string keys allowed"):
        deep = NestedDict(a={1: 1})
        deep.flatten()


# Common input feature naming conventions
@pytest.mark.parametrize(
    "sys_coords",
    [("x", "y"), ("x_1", "x_2"), ("x1", "x2"), ("ab", "uv")],
    ids=["simple", "underscore", "numbered", "arbitrary"],
)
def test_sindy_equations_to_sympy(sys_coords):
    f1, f2 = sys_coords
    model = ps.SINDy().fit(np.random.normal(size=(10, 2)), 1, feature_names=[f1, f2])
    opt = model.optimizer
    opt.coef_ = np.random.normal(size=opt.coef_.shape)
    _, sympy_eqs = _sindy_equations_to_sympy(model)
    coeff_dict = _sympy_expr_to_feat_coeff(sympy_eqs)
    expected_keys = {
        sp.Integer(1.0),
        sp.Symbol(f1),
        sp.Symbol(f2),
        sp.Mul(sp.Symbol(f1), sp.Symbol(f2)),
        sp.Pow(sp.Symbol(f1), 2),
        sp.Pow(sp.Symbol(f2), 2),
    }
    assert all(d.keys() == expected_keys for d in coeff_dict)


# Consider additional feature libraries (e.g. cos, sin) later
@pytest.mark.parametrize(
    "str_exprs, expected_coeff_dict",
    [
        pytest.param(["3"], {sp.Integer(1.0): 3.0}, id="single_const_term"),
        pytest.param(["3x"], {sp.Symbol("x"): 3.0}, id="single_term"),
        pytest.param(["3.1 x"], {sp.Symbol("x"): 3.1}, id="single_term_alt"),
        pytest.param(["z"], {sp.Symbol("z"): 1.0}, id="single_term_no_coeff"),
        pytest.param(
            ["4xy"],
            {sp.Mul(sp.Symbol("x"), sp.Symbol("y")): 4.0},
            id="single_term_mult",
        ),
        pytest.param(
            ["xy"],
            {sp.Mul(sp.Symbol("x"), sp.Symbol("y")): 1.0},
            id="single_term_mult_no_coeff",
        ),
        pytest.param(
            ["2*x*y"],
            {sp.Mul(sp.Symbol("x"), sp.Symbol("y")): 2.0},
            id="single_term_mult_alt",
        ),
        pytest.param(["3x**2"], {sp.Pow(sp.Symbol("x"), 2): 3.0}, id="single_term_pow"),
        pytest.param(
            ["x**2"], {sp.Pow(sp.Symbol("x"), 2): 1.0}, id="single_term_pow_no_coeff"
        ),
        pytest.param(
            ["3x + x**2"],
            {sp.Symbol("x"): 3.0, sp.Pow(sp.Symbol("x"), 2): 1.0},
            id="multiple_terms",
        ),
        pytest.param(
            ["x**2 + 3x"],
            {sp.Pow(sp.Symbol("x"), 2): 1.0, sp.Symbol("x"): 3.0},
            id="multiple_terms_reverse_order",
        ),
    ],
)
def test_sympy_expr_to_feat_coeff(str_exprs, expected_coeff_dict):
    sp_exprs = [
        sp.parse_expr(ex, transformations=XFORMS, evaluate=False) for ex in str_exprs
    ]
    result = _sympy_expr_to_feat_coeff(sp_exprs)
    assert result == [expected_coeff_dict]
    assert list(result[0]) == list(expected_coeff_dict)


def test_unionize_coeff_dicts_aligns_features():
    class DummyFeatures:
        def get_feature_names(
            self, input_features: Sequence = ["x", "y", "z"]
        ) -> list[str]:
            x, y, z = tuple(input_features)
            return ["1", x, y, z]

    class DummyModel:
        feature_names: list[str] = ["x", "y", "z"]
        feature_library: ps.feature_library.base.BaseFeatureLibrary = DummyFeatures()

        def equations(self, precision: int = 10):
            # Three coordinates:
            return ["x", "x + y", "x + y"]

    x, y, z = sp.symbols("x y z")
    true_equations = [
        {x: 2.0},
        {x: 2.0, z: 3.0},
        {x: 2.0, z: 3.0},
    ]

    true_aligned, est_aligned = unionize_coeff_dicts(DummyModel(), true_equations)

    # All coordinates should share the same feature keys, in order
    all_keys = list(true_aligned[0].keys())
    assert all(list(d.keys()) == all_keys for d in true_aligned)
    assert all(list(d.keys()) == all_keys for d in est_aligned)

    # The feature union should include x, y, and z
    assert all_keys == [1, x, y, z]


def test_coeff_dicts_to_matrix_basic():
    x, y = sp.symbols("x y")
    coeffs = [
        {x: 1.0, y: 2.0},
        {x: 3.0, y: 4.0},
    ]

    mat, feature_names = _coeff_dicts_to_matrix(coeffs)

    # Shape matches number of coordinates and features
    assert mat.shape == (2, 2)

    # Features are ordered by their string representation
    assert feature_names == ["x", "y"]

    # Rows correspond to the input dictionaries
    assert mat[0, :].tolist() == [1.0, 2.0]
    assert mat[1, :].tolist() == [3.0, 4.0]


@pytest.mark.parametrize(
    "rhs_name", ["lorenz", "rossler", "vanderpol", "sho", "cubicho", "kinematics"]
)
@pytest.mark.parametrize("array_namespace", ["numpy", "jax"])
def test_gen_data(rhs_name, array_namespace, jax_cpu_only):
    result = gen_data(
        rhs_name, t_end=0.1, noise_abs=0.01, seed=42, array_namespace=array_namespace
    )["data"]
    trajectories = result["trajectories"]
    assert len(trajectories) == 1
    traj = trajectories[0]
    assert traj.x_train.shape == traj.x_train_true_dot.shape
