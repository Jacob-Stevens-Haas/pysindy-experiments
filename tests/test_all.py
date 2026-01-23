import jax
import pytest
import sympy as sp

jax.config.update("jax_platform_name", "cpu")  # diffrax issue 722
# Alsorequires successive E402

from sindy_exp.data import gen_data  # noqa: E402
from sindy_exp.plotting import _coeff_dicts_to_matrix  # noqa: E402
from sindy_exp.typing import NestedDict  # noqa: E402
from sindy_exp.utils import unionize_coeff_dicts  # noqa: E402


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


def test_unionize_coeff_dicts_aligns_features():
    class DummyModel:
        def equations(self, precision: int = 10):
            # Two coordinates: x and x + y
            return ["x", "x + y"]

    x, y, z = sp.symbols("x y z")
    true_equations = [
        {x: 2.0},
        {x: 2.0, z: 3.0},
    ]

    true_aligned, est_aligned = unionize_coeff_dicts(DummyModel(), true_equations)

    # All coordinates should share the same feature keys
    all_keys = set(true_aligned[0].keys())
    assert all(set(d.keys()) == all_keys for d in true_aligned)
    assert all(set(d.keys()) == all_keys for d in est_aligned)

    # The feature union should include x, y, and z
    assert all_keys == {x, y, z}


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
