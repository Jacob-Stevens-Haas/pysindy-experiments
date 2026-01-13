import numpy as np
import pysindy as ps
import pytest

from sindy_exp.typing import NestedDict
from sindy_exp.utils import unionize_coeff_matrices


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


def test_unionize_coeff_matrices():
    model = ps.SINDy()
    data = np.arange(10)
    data = np.vstack((data, data)).T
    model.fit(data, 0.1, feature_names=["x", "y"])
    coeff_true = [{"y": -1.0, "zorp_x": 0.1}, {"x": 1.0, "zorp_y": 0.1}]
    true, est, feats = unionize_coeff_matrices(model, (["x", "y"], coeff_true))
    assert len(feats) == true.shape[1]
    assert len(feats) == est.shape[1]
    assert est.shape == true.shape


def test_unionize_coeff_matrices_translation():
    model = ps.SINDy()
    data = np.arange(10)
    data = np.vstack((data, data)).T
    model.fit(data, 0.1, feature_names=["a", "b"])
    coeff_true = [{"y": -1.0}, {"x": 1.0}]
    true, est, feats = unionize_coeff_matrices(model, (["x", "y"], coeff_true))
    assert len(feats) == true.shape[1]
    assert len(feats) == est.shape[1]
    assert est.shape == true.shape


def test_unionize_coeff_matrices_strict():
    model = ps.SINDy()
    data = np.arange(10)
    data = np.vstack((data, data)).T
    model.fit(data, 0.1, feature_names=["a", "b"])
    coeff_true = [{"y": -1.0}, {"x": 1.0}]
    with pytest.raises(ValueError, match="True model and fit model"):
        unionize_coeff_matrices(model, (["x", "y"], coeff_true), True)

def test_gen_data():
    from sindy_exp.data import gen_data

    data = gen_data("lorenz", noise_abs=0.01, seed=42)["data"]
    assert data.x_train[0].shape == data.x_dot_test[0].shape